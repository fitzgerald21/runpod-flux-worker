# Use an official RunPod PyTorch base image for compatibility and speed
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# --- Bake Model into Image ---
# Copy the downloader script into the container
COPY download_model.py .

# Run the script to download the model. This layer will be cached if the script doesn't change.
RUN python download_model.py

# --- Copy Application Code ---
# Copy the handler last, as it's the most frequently changed file
COPY handler.py .

# Command to run the worker script when the container starts
CMD ["python", "-u", "handler.py"]
