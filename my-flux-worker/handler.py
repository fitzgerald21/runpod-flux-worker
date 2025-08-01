import os
import base64
from io import BytesIO
import torch
import runpod
from PIL import Image
from diffusers import FluxKontextPipeline

# --- Global Scope: Model and Pipeline Initialization ---
# This code runs only once when a new worker instance starts.
PIPELINE = None

def initialize_pipeline():
    """
    Loads the model and pipeline into memory and moves it to the GPU.
    This function is called once per worker cold start.
    """
    global PIPELINE
    # Set a local cache directory within the container.
    # This is good practice for Hugging Face models.
    cache_dir = "/app/huggingface_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # The model identifier for the Nunchaku-quantized version of Flux Kontext.
    model_id = "nunchaku-tech/nunchaku-flux.1-kontext-dev"
    
    # Use bfloat16 for a good balance of performance and precision on modern GPUs.
    torch_dtype = torch.bfloat16

    print(f"Initializing pipeline for model: {model_id}...")
    try:
        pipe = FluxKontextPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir
        )
        pipe.to("cuda")
        PIPELINE = pipe
        print("Pipeline initialized successfully and moved to GPU.")
    except Exception as e:
        print(f"Fatal error during pipeline initialization: {e}")
        # If initialization fails, the worker cannot process jobs.
        # The error will be logged, and the worker may be terminated.
        PIPELINE = None

# --- Handler Function: Processing a Single Job ---
def handler(job):
    """
    This function is called for each incoming API request.
    It expects a job input containing a base64 encoded image and a text prompt.
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
        # Decode the base64 string into bytes, then open as a PIL Image
        image_bytes = base64.b64decode(base64_image)
        image_input = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to decode or open base64 image: {e}"}

    # --- Inference Parameters ---
    # Use provided parameters or fall back to sensible defaults.
    guidance_scale = job_input.get('guidance_scale', 5.0)
    num_inference_steps = job_input.get('num_inference_steps', 20)
    generator = torch.Generator(device="cuda").manual_seed(job_input.get('seed', 42))

    print(f"Running inference with prompt: '{prompt}'")
    
    # --- Inference Execution ---
    try:
        result_image = PIPELINE(
            prompt=prompt,
            image=image_input,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images
    except Exception as e:
        print(f"Error during model inference: {e}")
        return {"error": f"An error occurred during inference: {e}"}

    print("Inference completed successfully.")

    # --- Output Processing ---
    # Convert the output PIL Image back to a base64 string for JSON response.
    buffered = BytesIO()
    result_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # --- Return Result ---
    return {"image_base64": img_str}

# --- Start the Serverless Worker ---
# Initialize the pipeline first before starting the server.
initialize_pipeline()
# Register the handler function with the RunPod SDK.
runpod.serverless.start({"handler": handler})
