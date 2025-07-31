# Use the official Runpod PyTorch image as a base
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel

# Set the working directory
WORKDIR /

# Install system-level dependencies required for ComfyUI and some libraries
RUN apt-get update && \
    apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone the ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Set the working directory to ComfyUI
WORKDIR /ComfyUI

# Install Python dependencies for ComfyUI
RUN pip install --no-cache-dir -r requirements.txt

# --- Model Downloads ---
# Download all models in a single RUN command to improve layer caching and reduce timeout risks
RUN wget -O ./models/unet/flux1-dev-kontext_fp8_scaled.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/flux1-dev-kontext_fp8_scaled.safetensors && \
    wget -O ./models/clip/t5xxl_fp16.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/t5xxl_fp16.safetensors && \
    wget -O ./models/clip/clip_l.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/clip_l.safetensors && \
    wget -O ./models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/ae.safetensors

# Switch back to the root directory for the worker code
WORKDIR /

# Copy the worker source code and the workflow file
COPY ./src /src
COPY ./workflow_api.json /workflow_api.json

# Set the command to run the Runpod handler
CMD ["python", "-u", "/src/rp_handler.py"]
