# download_model.py
import os
from diffusers import FluxKontextPipeline

# --- Configuration ---
# The model we want to bake into our container image
MODEL_ID = "black-forest-labs/FLUX.1-kontext-dev-fp8"
# The directory where the model will be saved inside the container
CACHE_DIR = "/app/huggingface_cache"

def download_model():
    """
    Downloads the specified Hugging Face model to a local directory.
    """
    print(f"Creating cache directory at {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print(f"Starting download of model: {MODEL_ID}...")
    # This command downloads all necessary model components to the specified directory.
    FluxKontextPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR
    )
    print("Model download completed successfully.")

if __name__ == "__main__":
    download_model()
