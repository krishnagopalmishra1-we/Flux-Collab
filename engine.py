import os
# Fix for Windows MAX_PATH limit [Errno 22] when downloading large models
os.environ["HF_HOME"] = os.path.abspath("./hf_cache")

import io
import time
import gc
import torch
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIEngine")

class GenerationEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use bfloat16 for A100 / Modern GPUs, float16 for T4/older
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.dtype = torch.bfloat16
            self.precision_str = "bf16 (A100/H100 Optimized)"
        else:
            self.dtype = torch.float16
            self.precision_str = "fp16"

        self.current_model_id = None
        self.pipeline = None
        self.active_loras = []
        self.loaded_model_type = None  # 'flux' or 'sdxl'

        logger.info(f"Initialized Generation Engine on device: {self.device} ({self.precision_str})")

    def load_model(self, model_id: str = "black-forest-labs/FLUX.1-schnell", model_type: str = "flux", token: str = None):
        """
        Loads foundation model (FLUX.1 or SDXL) without safety filters.
        """
        if self.current_model_id == model_id and self.pipeline is not None:
            logger.info(f"Model {model_id} already loaded.")
            return True

        logger.info(f"Loading foundation model: {model_id} (Type: {model_type})...")
        
        # Free previous model memory
        if self.pipeline is not None:
            del self.pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            use_dtype = torch.bfloat16 if model_type.lower() == "flux" else self.dtype
            pretrained_kwargs = {
                "torch_dtype": use_dtype,
                "use_safetensors": True,
                "low_cpu_mem_usage": True,
            }
            if token:
                pretrained_kwargs["token"] = token

            if model_type.lower() == "flux":
                from diffusers import FluxPipeline
                self.pipeline = FluxPipeline.from_pretrained(
                    model_id, **pretrained_kwargs
                )
            elif model_type.lower() == "qwen":
                try:
                    from diffusers import QwenImage21Pipeline
                    pipe_cls = QwenImage21Pipeline
                except ImportError:
                    from diffusers import DiffusionPipeline
                    pipe_cls = DiffusionPipeline
                self.pipeline = pipe_cls.from_pretrained(
                    model_id, **pretrained_kwargs
                )
            elif model_type.lower() == "sd3":
                from diffusers import StableDiffusion3Pipeline
                self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                    model_id, **pretrained_kwargs
                )
            else:
                from diffusers import StableDiffusionXLPipeline, AutoencoderKL
                # Load SDXL without safety checker restrictions
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id, **pretrained_kwargs
                )

            # Ensure pipeline is transferred to GPU and disable safety checkers if present
            if hasattr(self.pipeline, "safety_checker"):
                self.pipeline.safety_checker = None
            if hasattr(self.pipeline, "requires_safety_checker"):
                self.pipeline.requires_safety_checker = False

            if self.device == "cuda":
                # Check VRAM for FLUX models to prevent OOM
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 20 and model_type.lower() == "flux":
                    logger.info(f"VRAM ({vram_gb:.1f}GB) is < 20GB. Using sequential CPU offload for FLUX.")
                    self.pipeline.enable_sequential_cpu_offload()
                else:
                    try:
                        self.pipeline.to("cuda")
                    except Exception as e:
                        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                            logger.warning("OOM when moving to CUDA, falling back to CPU offload")
                            self.pipeline.enable_model_cpu_offload()
                        else:
                            raise e
                # PyTorch 2.0+ SDPA is enabled by default

            self.current_model_id = model_id
            self.loaded_model_type = model_type.lower()
            self.active_loras = []
            logger.info(f"Successfully loaded {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise e

    def apply_loras(self, loras: list):
        """
        Applies dynamic LoRA weights.
        loras format: [{'repo_or_path': '...', 'weight_name': '...', 'scale': 0.8}]
        """
        if not self.pipeline:
            raise RuntimeError("No model loaded to apply LoRAs to.")

        try:
            # Unload existing adapter weights if any
            if hasattr(self.pipeline, "unload_lora_weights"):
                try:
                    self.pipeline.unload_lora_weights()
                except Exception:
                    pass

            adapter_names = []
            adapter_weights = []

            for idx, lora in enumerate(loras):
                path = lora.get("repo_or_path")
                weight_name = lora.get("weight_name", None)
                scale = float(lora.get("scale", 1.0))

                if not path:
                    continue

                adapter_name = f"lora_{idx}"
                logger.info(f"Loading LoRA [{adapter_name}]: {path} (scale: {scale})")

                hf_token = os.environ.get("HF_TOKEN", None)
                if weight_name:
                    self.pipeline.load_lora_weights(path, weight_name=weight_name, adapter_name=adapter_name, token=hf_token)
                else:
                    self.pipeline.load_lora_weights(path, adapter_name=adapter_name, token=hf_token)

                adapter_names.append(adapter_name)
                adapter_weights.append(scale)

            if adapter_names and hasattr(self.pipeline, "set_adapters"):
                self.pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
                logger.info(f"Active LoRA Adapters set: {adapter_names} with weights {adapter_weights}")

            self.active_loras = loras
            return True
        except Exception as e:
            logger.error(f"Error applying LoRAs: {str(e)}")
            raise e

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        mode: str = "t2i",
        init_image: Image.Image = None,
        denoising_strength: float = 0.75,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = -1,
        loras: list = None
    ):
        if not self.pipeline:
            # Auto fallback load black-forest-labs/FLUX.1-schnell if not loaded
            self.load_model("black-forest-labs/FLUX.1-schnell", "flux")

        if loras:
            self.apply_loras(loras)

        # Handle Random / Locked Seed
        if seed is None or seed < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(device=self.device).manual_seed(seed)

        start_time = time.time()
        logger.info(f"Starting generation | Seed: {seed} | Resolution: {width}x{height} | Steps: {num_inference_steps}")

        if torch.cuda.is_available():
            vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_total_gb < 20 and self.loaded_model_type == "flux":
                logger.warning("VRAM < 20GB detected for FLUX. CPU offloading will likely be needed.")

        kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "generator": generator
        }

        # Model specific parameters
        if self.loaded_model_type == "flux":
            kwargs["guidance_scale"] = 0.0 if "schnell" in self.current_model_id.lower() else guidance_scale
            kwargs["max_sequence_length"] = 256
        else: # SDXL
            kwargs["guidance_scale"] = guidance_scale
            if negative_prompt:
                kwargs["negative_prompt"] = negative_prompt

        # Image-to-Image setup
        if mode == "i2i" and init_image is not None:
            # Most Img2Img diffusers pipelines infer resolution from the image and crash if width/height is explicitly passed
            kwargs.pop("width", None)
            kwargs.pop("height", None)
            kwargs["image"] = init_image.resize((width, height))
            kwargs["strength"] = denoising_strength
            # Convert to Img2Img pipeline dynamically if SDXL
            if self.loaded_model_type == "sdxl":
                from diffusers import StableDiffusionXLImg2ImgPipeline
                i2i_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(self.pipeline)
                output = i2i_pipe(**kwargs)
            elif self.loaded_model_type == "flux":
                try:
                    from diffusers import FluxImg2ImgPipeline
                    i2i_pipe = FluxImg2ImgPipeline.from_pipe(self.pipeline)
                    output = i2i_pipe(**kwargs)
                except ImportError:
                    logger.warning("FluxImg2ImgPipeline not found in this diffusers version. Falling back to standard pipeline.")
                    output = self.pipeline(**kwargs)
            else:
                # Fallback for others
                output = self.pipeline(**kwargs)
        else:
            output = self.pipeline(**kwargs)

        generation_time = round(time.time() - start_time, 2)
        generated_image = output.images[0]

        logger.info(f"Generation completed in {generation_time}s")

        meta = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "model": self.current_model_id,
            "generation_time": generation_time,
            "device": self.device,
            "precision": self.precision_str
        }

        return generated_image, meta

    def get_system_info(self):
        vram_total = 0
        vram_free = 0
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            vram_free = round(torch.cuda.mem_get_info()[0] / (1024**3), 2)
        else:
            gpu_name = "CPU Only"

        return {
            "gpu_name": gpu_name,
            "vram_total_gb": vram_total,
            "vram_free_gb": vram_free,
            "device": self.device,
            "precision": self.precision_str,
            "loaded_model": self.current_model_id
        }
