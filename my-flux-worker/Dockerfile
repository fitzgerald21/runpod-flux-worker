# Use a currently available and stable NVIDIA PyTorch image as a base
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

WORKDIR /

# --- Step 1: Install System Dependencies ---
RUN echo "Starting Step 1: Installing system dependencies with apt-get" && \
    apt-get update && \
    apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "Finished Step 1."

# --- Step 2: Clone THIS repository to get the worker code ---
# This bypasses any issues with the Runpod build context by cloning the repo internally.
RUN echo "Starting Step 2: Cloning worker repository to get source files" && \
    git clone https://github.com/fitzgerald21/runpod-flux-worker.git /repo && \
    echo "Finished Step 2."

# --- Step 3: Copy worker code from the cloned repo ---
# This copies the files from the internal clone, not the faulty build context.
RUN echo "Starting Step 3: Copying worker code from internal repo clone" && \
    cp -r /repo/src /src && \
    cp /repo/workflow_api.json /workflow_api.json && \
    echo "Finished Step 3."

# --- Step 4: Clone ComfyUI Repository ---
RUN echo "Starting Step 4: Cloning ComfyUI repository" && \
    git clone https://github.com/comfyanonymous/ComfyUI.git && \
    echo "Finished Step 4."

# Set the working directory to ComfyUI for its setup
WORKDIR /ComfyUI

# --- Step 5: Install Python Dependencies ---
RUN echo "Starting Step 5: Installing Python requirements for ComfyUI" && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "Finished Step 5."

# --- Step 6: Download Models ---
RUN echo "Starting Step 6: Downloading models" && \
    wget -O ./models/unet/flux1-dev-kontext_fp8_scaled.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/flux1-dev-kontext_fp8_scaled.safetensors && \
    wget -O ./models/clip/t5xxl_fp16.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/t5xxl_fp16.safetensors && \
    wget -O ./models/clip/clip_l.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/clip_l.safetensors && \
    wget -O ./models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/ae.safetensors && \
    echo "Finished Step 6."

# --- Step 7: Set the final command ---
# The handler script is at /src/rp_handler.py, which is where we copied it in Step 3.
CMD ["python", "-u", "/src/rp_handler.py"]
