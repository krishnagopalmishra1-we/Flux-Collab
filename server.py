import os
import io
import json
import base64
import threading
import traceback
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

HF_TOKEN = os.environ.get('HF_TOKEN', None)

from engine import GenerationEngine

app = FastAPI(title="LuxGen AI - Unrestricted High-End Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

# Setup static files and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
templates = Jinja2Templates(directory="templates")

# Global Engine instance
engine = GenerationEngine()
generation_lock = threading.Lock()

class LoRAModel(BaseModel):
    repo_or_path: str
    weight_name: Optional[str] = None
    scale: float = 1.0

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    mode: str = "t2i"  # 't2i' or 'i2i'
    denoising_strength: Optional[float] = 0.75
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    num_inference_steps: Optional[int] = 25
    guidance_scale: Optional[float] = 4.0
    seed: Optional[int] = -1
    model_id: Optional[str] = "black-forest-labs/FLUX.1-schnell"
    model_type: Optional[str] = "flux"  # supported model_types: 'flux', 'sdxl', 'sd3', 'qwen'
    loras: Optional[List[LoRAModel]] = []
    init_image_base64: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def serve_studio(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/system")
async def get_system_status():
    return engine.get_system_info()

@app.post("/api/load_model")
def load_model_endpoint(payload: dict):
    model_id = payload.get("model_id", "black-forest-labs/FLUX.1-schnell")
    model_type = payload.get("model_type", "flux")
    try:
        if HF_TOKEN:
            engine.load_model(model_id, model_type, token=HF_TOKEN)
        else:
            engine.load_model(model_id, model_type)
        return {"status": "success", "message": f"Loaded model {model_id}"}
    except Exception as e:
        tb = traceback.format_exc()
        print("--- LOAD MODEL EXCEPTION ---")
        print(tb)
        with open("error.log", "w") as f:
            f.write(tb)
        raise HTTPException(status_code=500, detail=f"Load Model Error: {str(e)} | Traceback: {tb}")

@app.post("/api/generate")
def generate_image(req: GenerationRequest):
    try:
        # Load model if different
        if engine.current_model_id != req.model_id:
            if HF_TOKEN:
                engine.load_model(req.model_id, req.model_type, token=HF_TOKEN)
            else:
                engine.load_model(req.model_id, req.model_type)

        # Parse Init Image for I2I if present
        init_img = None
        if req.mode == "i2i" and req.init_image_base64:
            try:
                img_data = base64.b64decode(req.init_image_base64.split(",")[-1])
                init_img = Image.open(io.BytesIO(img_data)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

        loras_dict = [lora.model_dump() for lora in req.loras] if req.loras else []

        with generation_lock:
            img, meta = engine.generate(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                mode=req.mode,
                init_image=init_img,
                denoising_strength=req.denoising_strength,
                width=req.width,
                height=req.height,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                loras=loras_dict
            )

        # Save output image
        filename = f"gen_{meta['seed']}_{int(meta['generation_time']*100)}.png"
        filepath = os.path.join("outputs", filename)
        img.save(filepath, format="PNG")

        # Convert image to Base64 data URL for instant render
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "image_url": f"/outputs/{filename}",
            "image_base64": f"data:image/png;base64,{img_str}",
            "meta": meta
        }

    except Exception as e:
        tb = traceback.format_exc()
        print("--- GENERATE EXCEPTION ---")
        print(tb)
        with open("error.log", "w") as f:
            f.write(tb)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)} | Traceback: {tb}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
