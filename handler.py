import os
import base64
from io import BytesIO
import torch
import runpod
from PIL import Image
from diffusers import FluxKontextPipeline
import bitsandbytes

# --- Global Scope: Model and Pipeline Initialization ---
PIPELINE = None
# Define the local path where the model was saved during the build
LOCAL_MODEL_PATH = "/app/huggingface_cache"

def initialize_pipeline():
    """
    Loads the pre-downloaded model from the local filesystem into memory.
    """
    global PIPELINE
    
    print(f"Initializing pipeline from local path: {LOCAL_MODEL_PATH}...")
    try:
        # Check if the model path exists
        if not os.path.exists(LOCAL_MODEL_PATH):
             raise RuntimeError(f"Model path not found: {LOCAL_MODEL_PATH}. The model was not baked into the image correctly.")

        pipe = FluxKontextPipeline.from_pretrained(
            LOCAL_MODEL_PATH,
            local_files_only=True # This is crucial, it prevents any network access
        )
        pipe.to("cuda")
        PIPELINE = pipe
        print("Pipeline initialized successfully from local files and moved to GPU.")
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
