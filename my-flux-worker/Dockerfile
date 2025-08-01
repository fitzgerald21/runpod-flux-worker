# Use a currently available and stable NVIDIA PyTorch image as a base
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# --- Step 1: Copy worker code FIRST ---
# This is the standard and most reliable way to include your code.
# It relies on the Runpod build context being set correctly.
WORKDIR /
COPY ./src /src
COPY ./workflow_api.json /workflow_api.json

# --- Step 2: Install System Dependencies ---
RUN apt-get update && \
    apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Step 3: Clone ComfyUI Repository ---
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Set the working directory to ComfyUI for its setup
WORKDIR /ComfyUI

# --- Step 4: Install Python Dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Step 5: Download Models ---
RUN wget -O ./models/unet/flux1-dev-kontext_fp8_scaled.safensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/flux1-dev-kontext_fp8_scaled.safensors && \
    wget -O ./models/clip/t5xxl_fp16.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/t5xxl_fp16.safetensors && \
    wget -O ./models/clip/clip_l.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/clip_l.safetensors && \
    wget -O ./models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/ae.safetensors

# --- Step 6: Set the final command ---
# The handler script is at /src/rp_handler.py, which we copied in Step 1.
CMD ["python", "-u", "/src/rp_handler.py"]
