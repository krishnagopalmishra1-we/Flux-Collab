document.addEventListener('DOMContentLoaded', () => {
    // --- State Variables ---
    let currentMode = 't2i'; // 't2i' or 'i2i'
    let selectedWidth = 1024;
    let selectedHeight = 1024;
    let initImageBase64 = null;
    let galleryItems = [];
    let timerInterval = null;
    let startTime = 0;
    let loraCounter = 0;

    // --- DOM Element References ---
    const modeT2iBtn = document.getElementById('mode-t2i');
    const modeI2iBtn = document.getElementById('mode-i2i');
    const modelSelect = document.getElementById('model-select');
    const styleSelect = document.getElementById('style-select');
    const promptInput = document.getElementById('prompt-input');
    const negativePromptInput = document.getElementById('negative-prompt');
    const negPromptGroup = document.getElementById('neg-prompt-group');
    const i2iDropzoneGroup = document.getElementById('i2i-dropzone-group');
    const initImageInput = document.getElementById('init-image-input');
    const dropzone = document.getElementById('dropzone');
    const dropzoneContent = document.getElementById('dropzone-content');
    const initImagePreview = document.getElementById('init-image-preview');
    const denoisingSlider = document.getElementById('denoising-slider');
    const valDenoising = document.getElementById('val-denoising');
    
    const aspectBtns = document.querySelectorAll('.aspect-btn');
    const stepsSlider = document.getElementById('steps-slider');
    const valSteps = document.getElementById('val-steps');
    const cfgSlider = document.getElementById('cfg-slider');
    const valCfg = document.getElementById('val-cfg');
    const seedInput = document.getElementById('seed-input');
    const btnRandomSeed = document.getElementById('btn-random-seed');
    
    const btnGenerate = document.getElementById('btn-generate');
    const genSpinner = document.getElementById('gen-spinner');
    const btnEnhancePrompt = document.getElementById('btn-enhance-prompt');

    // Canvas & Gallery
    const placeholderState = document.getElementById('placeholder-state');
    const loadingState = document.getElementById('loading-state');
    const generatedImage = document.getElementById('generated-image');
    const loadingTimer = document.getElementById('loading-timer');
    const loadingStatusText = document.getElementById('loading-status-text');
    const metadataBar = document.getElementById('metadata-bar');
    const metaSeed = document.getElementById('meta-seed');
    const metaTime = document.getElementById('meta-time');
    const metaRes = document.getElementById('meta-res');
    const metaSteps = document.getElementById('meta-steps');
    
    const btnDownload = document.getElementById('btn-download');
    const btnFullscreen = document.getElementById('btn-fullscreen');
    const galleryContainer = document.getElementById('gallery-container');
    const galleryCount = document.getElementById('gallery-count');

    // System Stats
    const statGpu = document.getElementById('stat-gpu');
    const statVram = document.getElementById('stat-vram');
    const statPrecision = document.getElementById('stat-precision');
    const btnRefreshSys = document.getElementById('btn-refresh-sys');

    // Accordions & LoRA
    const advAccordionToggle = document.getElementById('adv-accordion-toggle');
    const advAccordionBody = document.getElementById('adv-accordion-body');
    const loraAccordionToggle = document.getElementById('lora-accordion-toggle');
    const loraAccordionBody = document.getElementById('lora-accordion-body');
    const btnAddLora = document.getElementById('btn-add-lora');
    const loraListContainer = document.getElementById('lora-list-container');
    const loraCountSpan = document.getElementById('lora-count');

    // Lightbox
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const lightboxClose = document.getElementById('lightbox-close');
    const lightboxBackdrop = document.getElementById('lightbox-backdrop');

    // --- System Status Fetcher ---
    async function fetchSystemStats() {
        try {
            const res = await fetch('/api/system');
            if (res.ok) {
                const data = await res.json();
                statGpu.textContent = data.gpu_name;
                statVram.textContent = `VRAM: ${data.vram_free_gb} / ${data.vram_total_gb} GB`;
                statPrecision.textContent = `Precision: ${data.precision}`;
            }
        } catch (e) {
            statGpu.textContent = "Colab Server Ready";
        }
    }
    fetchSystemStats();
    if (btnRefreshSys) btnRefreshSys.addEventListener('click', fetchSystemStats);

    // --- Mode Toggles ---
    modeT2iBtn.addEventListener('click', () => switchMode('t2i'));
    modeI2iBtn.addEventListener('click', () => switchMode('i2i'));

    function switchMode(mode) {
        currentMode = mode;
        if (mode === 't2i') {
            modeT2iBtn.classList.add('active');
            modeI2iBtn.classList.remove('active');
            i2iDropzoneGroup.classList.add('hidden');
        } else {
            modeI2iBtn.classList.add('active');
            modeT2iBtn.classList.remove('active');
            i2iDropzoneGroup.classList.remove('hidden');
        }
    }

    // --- Model Change listener (adapt parameters per model type) ---
    const customModelInput = document.getElementById('custom-model-input');

    modelSelect.addEventListener('change', (e) => {
        const selectedOption = e.target.options[e.target.selectedIndex];
        const isCustom = selectedOption.value === 'custom';

        // Show/hide custom model text input
        if (customModelInput) {
            customModelInput.classList.toggle('hidden', !isCustom);
        }

        // Auto-adjust sliders from HTML data attributes
        const optSteps = selectedOption.getAttribute('data-steps');
        const optCfg = selectedOption.getAttribute('data-cfg');
        
        if (optSteps) {
            stepsSlider.value = optSteps;
            valSteps.textContent = optSteps;
        }
        if (optCfg) {
            cfgSlider.value = optCfg;
            valCfg.textContent = optCfg;
        }
    });

    // --- Image-to-Image Dropzone & File Upload ---
    dropzone.addEventListener('click', () => initImageInput.click());
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--accent-primary)';
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = 'var(--input-border)';
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--input-border)';
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleInitFile(e.dataTransfer.files[0]);
        }
    });

    initImageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleInitFile(e.target.files[0]);
        }
    });

    function handleInitFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            initImageBase64 = e.target.result;
            initImagePreview.src = initImageBase64;
            initImagePreview.classList.remove('hidden');
            dropzoneContent.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    // Sliders
    denoisingSlider.addEventListener('input', (e) => valDenoising.textContent = e.target.value);
    stepsSlider.addEventListener('input', (e) => valSteps.textContent = e.target.value);
    cfgSlider.addEventListener('input', (e) => valCfg.textContent = e.target.value);

    // Aspect Ratio Toggles
    aspectBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            aspectBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedWidth = parseInt(btn.getAttribute('data-w'));
            selectedHeight = parseInt(btn.getAttribute('data-h'));
        });
    });

    // Seed Randomizer
    btnRandomSeed.addEventListener('click', () => seedInput.value = -1);

    // Prompt Enhancer Assistant (LLM Powered)
    btnEnhancePrompt.addEventListener('click', async () => {
        const cur = promptInput.value.trim();
        if (!cur) {
            alert('Please enter a basic prompt first to enhance it!');
            return;
        }
        
        const originalText = btnEnhancePrompt.innerHTML;
        btnEnhancePrompt.innerHTML = '<i class="lucide lucide-loader animate-spin"></i> Enhancing...';
        btnEnhancePrompt.disabled = true;

        try {
            const res = await fetch('/api/enhance_prompt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: cur })
            });
            const data = await res.json();
            if (res.ok && data.enhanced_prompt) {
                promptInput.value = data.enhanced_prompt;
            } else {
                throw new Error(data.detail || 'API returned failure');
            }
        } catch (err) {
            console.warn('Enhancement API failed, using local fallback:', err);
            // Local fallback logic if HF API is overloaded/down
            const modifiers = [
                "masterpiece, highly detailed 8k resolution, cinematic lighting, photorealistic, intricate textures, volumetric depth, professional shot",
                "hyper-realistic studio portrait, award-winning photography, crisp focus, soft rim lighting, subsurface scattering",
                "epic conceptual art, ultra-detailed, vibrant color grading, dynamic composition, masterpiece"
            ];
            const randomMod = modifiers[Math.floor(Math.random() * modifiers.length)];
            promptInput.value = `${cur}, ${randomMod}`;
        } finally {
            btnEnhancePrompt.innerHTML = originalText;
            btnEnhancePrompt.disabled = false;
        }
    });

    // Accordions
    advAccordionToggle.addEventListener('click', () => {
        advAccordionToggle.classList.toggle('active');
        advAccordionBody.classList.toggle('hidden');
    });

    loraAccordionToggle.addEventListener('click', () => {
        loraAccordionToggle.classList.toggle('active');
        loraAccordionBody.classList.toggle('hidden');
    });

    // --- Dynamic LoRA Manager ---
    btnAddLora.addEventListener('click', () => {
        loraCounter++;
        const row = document.createElement('div');
        row.className = 'lora-row';
        row.id = `lora-row-${loraCounter}`;
        row.innerHTML = `
            <input type="text" class="custom-input lora-input lora-path" placeholder="HuggingFace Repo ID or Path (e.g. XLabs-AI/flux-lora-collection)">
            <input type="number" class="custom-input lora-weight-input lora-scale" value="0.8" step="0.1" min="0.0" max="2.0" title="LoRA Scale Weight">
            <button class="btn btn-secondary btn-icon btn-sm remove-lora-btn" onclick="removeLoraRow(${loraCounter})">
                <i data-lucide="trash-2"></i>
            </button>
        `;
        loraListContainer.appendChild(row);
        updateLoraCount();
        if (window.lucide) lucide.createIcons();
    });

    window.removeLoraRow = function(id) {
        const r = document.getElementById(`lora-row-${id}`);
        if (r) r.remove();
        updateLoraCount();
    };

    function updateLoraCount() {
        const count = loraListContainer.querySelectorAll('.lora-row').length;
        loraCountSpan.textContent = count;
    }

    function collectLoras() {
        const rows = loraListContainer.querySelectorAll('.lora-row');
        const list = [];
        rows.forEach(r => {
            const path = r.querySelector('.lora-path').value.trim();
            const scale = parseFloat(r.querySelector('.lora-scale').value) || 1.0;
            if (path) {
                list.push({ repo_or_path: path, scale: scale });
            }
        });
        return list;
    }

    // --- Style Presets ---
    const STYLE_PRESETS = {
        base: { pos: "", neg: "" },
        photorealistic: {
            pos: "masterpiece, ultra-realistic, 8k resolution, raw photo, highly detailed, sharp focus, professional photography",
            neg: "illustration, painting, cartoon, 3d, cg, deformed, blurry, ugly, sketch, low quality"
        },
        cinematic: {
            pos: "cinematic lighting, dramatic depth of field, movie still, epic composition, color graded",
            neg: "amateur, badly directed, poor lighting, standard, uninspired, flat, dull"
        },
        anime: {
            pos: "anime artwork, studio ghibli style, makoto shinkai, colorful, highly detailed anime, masterpiece",
            neg: "photo, realism, ugly, 3d render, lowres, bad anatomy, bad hands, text, error"
        },
        digital_art: {
            pos: "concept art, trending on artstation, digital illustration, highly detailed, vibrant, beautiful",
            neg: "photograph, realistic, messy, low quality, artifact, jpeg"
        },
        fantasy: {
            pos: "ethereal fantasy concept art, magical, highly detailed, intricate, dnd art, masterpiece",
            neg: "sci-fi, modern, ordinary, mundane, poor quality, bad anatomy"
        },
        "3d_render": {
            pos: "octane render, unreal engine 5, ray tracing, incredibly detailed 3d, cinematic 3d",
            neg: "2d, flat, illustration, painting, photo, unshaded, jagged"
        }
    };

    // Make Style Select instantly update the UI so users know it's working
    styleSelect.addEventListener('change', (e) => {
        const selectedStyle = e.target.value;
        if (STYLE_PRESETS[selectedStyle] && selectedStyle !== 'base') {
            const curP = promptInput.value.trim();
            const curN = negativePromptInput.value.trim();
            
            // Check if it already has tags to avoid spamming
            const firstPosTag = STYLE_PRESETS[selectedStyle].pos.split(',')[0];
            if (!curP.includes(firstPosTag)) {
                promptInput.value = curP ? `${curP}, ${STYLE_PRESETS[selectedStyle].pos}` : STYLE_PRESETS[selectedStyle].pos;
            }
            
            const firstNegTag = STYLE_PRESETS[selectedStyle].neg.split(',')[0];
            if (!curN.includes(firstNegTag)) {
                negativePromptInput.value = curN ? `${curN}, ${STYLE_PRESETS[selectedStyle].neg}` : STYLE_PRESETS[selectedStyle].neg;
            }
        }
    });

    // --- Main Generation Handler ---
    btnGenerate.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert('Please enter a prompt description before generating.');
            return;
        }

        // Set UI Loading State
        setGeneratingState(true);

        const selectedOption = modelSelect.options[modelSelect.selectedIndex];
        let modelId = selectedOption.value;
        let modelType = selectedOption.getAttribute('data-type');

        // Handle custom model input
        if (modelId === 'custom' && customModelInput) {
            modelId = customModelInput.value.trim();
            modelType = 'flux'; // default type for custom models
            if (!modelId) {
                alert('Please enter a HuggingFace Model ID for the custom model.');
                setGeneratingState(false);
                return;
            }
        }

        // Apply Style Presets (Handled at UI level now, see below, but keep here as fallback)
        const selectedStyle = styleSelect.value;
        let finalPrompt = prompt;
        let finalNegative = negativePromptInput.value.trim();

        if (STYLE_PRESETS[selectedStyle] && selectedStyle !== 'base') {
            // We only inject if it's not already in the prompt (prevent doubling if UI already added it)
            if (!finalPrompt.includes(STYLE_PRESETS[selectedStyle].pos.split(',')[0])) {
                finalPrompt = `${prompt}, ${STYLE_PRESETS[selectedStyle].pos}`;
            }
            if (!finalNegative.includes(STYLE_PRESETS[selectedStyle].neg.split(',')[0])) {
                finalNegative = finalNegative ? `${finalNegative}, ${STYLE_PRESETS[selectedStyle].neg}` : STYLE_PRESETS[selectedStyle].neg;
            }
        }

        const payload = {
            prompt: finalPrompt,
            negative_prompt: finalNegative,
            mode: currentMode,
            denoising_strength: parseFloat(denoisingSlider.value),
            width: selectedWidth,
            height: selectedHeight,
            num_inference_steps: parseInt(stepsSlider.value),
            guidance_scale: parseFloat(cfgSlider.value),
            seed: parseInt(seedInput.value),
            model_id: modelId,
            model_type: modelType,
            loras: collectLoras(),
            init_image_base64: currentMode === 'i2i' ? initImageBase64 : null
        };

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Generation failed on server');
            }

            const data = await response.json();

            // Render Output Image
            generatedImage.src = data.image_base64;
            generatedImage.classList.remove('hidden');
            loadingState.classList.add('hidden');
            placeholderState.classList.add('hidden');

            // Metadata update
            metaSeed.textContent = data.meta.seed;
            metaTime.textContent = data.meta.generation_time;
            metaRes.textContent = `${data.meta.width}x${data.meta.height}`;
            metaSteps.textContent = data.meta.steps;
            metadataBar.classList.remove('hidden');

            btnDownload.disabled = false;
            btnFullscreen.disabled = false;

            // Save to Gallery
            addToGallery(data.image_base64, data.meta);
            fetchSystemStats();

        } catch (error) {
            alert(`Generation Error: ${error.message}`);
            placeholderState.classList.remove('hidden');
            loadingState.classList.add('hidden');
        } finally {
            setGeneratingState(false);
        }
    });

    function setGeneratingState(isGen) {
        btnGenerate.disabled = isGen;
        if (isGen) {
            placeholderState.classList.add('hidden');
            generatedImage.classList.add('hidden');
            loadingState.classList.remove('hidden');
            genSpinner.classList.remove('hidden');
            
            // Clear any orphaned timer before starting a new one
            if (timerInterval) clearInterval(timerInterval);
            startTime = Date.now();
            loadingTimer.textContent = '0.0s';
            timerInterval = setInterval(() => {
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                loadingTimer.textContent = `${elapsed}s`;
            }, 100);
        } else {
            genSpinner.classList.add('hidden');
            if (timerInterval) {
                clearInterval(timerInterval);
                timerInterval = null;
            }
        }
    }

    // --- Gallery System (BUG-09 FIX: Capped at 30 items to prevent memory leak) ---
    const MAX_GALLERY_ITEMS = 30;

    function addToGallery(imgSrc, meta) {
        // Convert base64 to Blob URL to reduce memory footprint
        let displaySrc = imgSrc;
        try {
            const byteStr = atob(imgSrc.split(',')[1]);
            const ab = new ArrayBuffer(byteStr.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteStr.length; i++) ia[i] = byteStr.charCodeAt(i);
            const blob = new Blob([ab], { type: 'image/png' });
            displaySrc = URL.createObjectURL(blob);
        } catch (e) { /* fallback to base64 if conversion fails */ }

        galleryItems.unshift({ imgSrc: displaySrc, meta });

        // Evict oldest items beyond cap
        while (galleryItems.length > MAX_GALLERY_ITEMS) {
            const removed = galleryItems.pop();
            if (removed.imgSrc.startsWith('blob:')) URL.revokeObjectURL(removed.imgSrc);
        }

        galleryCount.textContent = `${galleryItems.length} items`;

        const thumb = document.createElement('img');
        thumb.src = displaySrc;
        thumb.className = 'gallery-thumb';
        thumb.title = `Seed: ${meta.seed}`;
        thumb.addEventListener('click', () => {
            generatedImage.src = displaySrc;
            metaSeed.textContent = meta.seed;
            metaTime.textContent = meta.generation_time;
            metaRes.textContent = `${meta.width}x${meta.height}`;
            metaSteps.textContent = meta.steps;
        });

        if (galleryContainer.querySelector('.empty-gallery')) {
            galleryContainer.innerHTML = '';
        }

        // Remove oldest DOM thumbnails beyond cap
        const existingThumbs = galleryContainer.querySelectorAll('.gallery-thumb');
        if (existingThumbs.length >= MAX_GALLERY_ITEMS) {
            existingThumbs[existingThumbs.length - 1].remove();
        }

        galleryContainer.prepend(thumb);
    }

    // --- Action Controls: Download & Fullscreen ---
    btnDownload.addEventListener('click', () => {
        if (!generatedImage.src) return;
        const a = document.createElement('a');
        a.href = generatedImage.src;
        a.download = `AURA_FLUX_${metaSeed.textContent}_${Date.now()}.png`;
        a.click();
    });

    btnFullscreen.addEventListener('click', () => {
        if (!generatedImage.src) return;
        lightboxImg.src = generatedImage.src;
        lightbox.classList.remove('hidden');
    });

    lightboxClose.addEventListener('click', () => lightbox.classList.add('hidden'));
    lightboxBackdrop.addEventListener('click', () => lightbox.classList.add('hidden'));

    // Keyboard ESC to close lightbox
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !lightbox.classList.contains('hidden')) {
            lightbox.classList.add('hidden');
        }
    });
});
