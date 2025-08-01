import os
import base64
from io import BytesIO
import torch
import runpod
from PIL import Image
from diffusers import FluxKontextPipeline
import bitsandbytes # Although not explicitly called, it's needed for fp8

# --- Global Scope: Model and Pipeline Initialization ---
PIPELINE = None

def initialize_pipeline():
    """
    Loads the model and pipeline into memory and moves it to the GPU.
    This function is called once per worker cold start.
    """
    global PIPELINE
    cache_dir = "/app/huggingface_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Correct model ID for the FP8 version
    model_id = "black-forest-labs/FLUX.1-kontext-dev-fp8"
    
    print(f"Initializing pipeline for model: {model_id}...")
    try:
        # Load the pipeline. For FP8 models, we don't need to specify torch_dtype.
        pipe = FluxKontextPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        pipe.to("cuda")
        PIPELINE = pipe
        print("Pipeline initialized successfully and moved to GPU.")
    except Exception as e:
        print(f"Fatal error during pipeline initialization: {e}")
        PIPELINE = None

# --- Handler Function: Processing a Single Job ---
def handler(job):
    """
    This function is called for each incoming API request.
    """
    if PIPELINE is None:
        return {
            "error": "Pipeline is not initialized. The worker failed to start correctly."
        }

    job_input = job.get('input', {})
    print(f"Received job: {job['id']}")

    # --- Input Validation ---
    base64_image = job_input.get('image_base64')
    prompt = job_input.get('prompt')

    if not base64_image or not prompt:
        return {"error": "Missing required inputs: 'image_base64' and 'prompt' are required."}

    # --- Input Processing ---
    try:
        image_bytes = base64.b64decode(base64_image)
        image_input = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to decode or open base64 image: {e}"}

    # --- Inference Parameters ---
    guidance_scale = job_input.get('guidance_scale', 5.0)
    num_inference_steps = job_input.get('num_inference_steps', 20)
    seed = job_input.get('seed', 42)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Get width/height and resize the input image
    width = job_input.get('width', 1024)
    height = job_input.get('height', 1024)
    
    print(f"Resizing input image to {width}x{height}")
    image_input = image_input.resize((width, height), Image.Resampling.LANCZOS)


    print(f"Running inference with prompt: '{prompt}'")
    
    # --- Inference Execution ---
    try:
        result_images = PIPELINE(
            prompt=prompt,
            image=image_input,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images
        
        # The pipeline returns a list of images, we'll take the first one.
        result_image = result_images[0]

    except Exception as e:
        print(f"Error during model inference: {e}")
        return {"error": f"An error occurred during inference: {e}"}

    print("Inference completed successfully.")

    # --- Output Processing ---
    buffered = BytesIO()
    result_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image_base64": img_str}

# --- Start the Serverless Worker ---
initialize_pipeline()
runpod.serverless.start({"handler": handler})
