import runpod
import json
import os
import base64
from io import BytesIO
from PIL import Image

# Import the runner class we created
from comfy_runner import ComfyRunner

# Initialize the ComfyRunner.
# This will start the ComfyUI server in the background.
runner = ComfyRunner()

def handler(job):
    """
    This is the main handler function that Runpod will call.

    Args:
        job (dict): A dictionary containing the job input.

    Returns:
        dict: A dictionary containing the output image (base64) or an error.
    """
    job_input = job.get('input', {})

    # --- Validate Inputs ---
    prompt = job_input.get('prompt')
    image_b64 = job_input.get('image')

    if not prompt:
        return {'error': 'A text prompt is required.'}
    if not image_b64:
        return {'error': 'An input image (base64 encoded) is required.'}

    # --- Prepare Input Image ---
    try:
        # Decode the base64 image
        image_data = base64.b64decode(image_b64)
        # Use PIL to open the image data
        image = Image.open(BytesIO(image_data))
        
        # Define the path to save the input image
        input_dir = "/ComfyUI/input"
        os.makedirs(input_dir, exist_ok=True)
        input_image_path = os.path.join(input_dir, "input_image.png")
        
        # Save the image in a format ComfyUI can read
        image.save(input_image_path)
        print(f"Input image saved to: {input_image_path}")
        
    except Exception as e:
        print(f"Error processing input image: {e}")
        return {'error': f'Failed to process input image: {str(e)}'}

    # --- Load and Modify Workflow ---
    try:
        with open('/workflow_api.json', 'r') as f:
            workflow = json.load(f)

        # Find the prompt node (ID 9) and set the text
        workflow['9']['inputs']['text'] = prompt

        # Find the LoadImage node (ID 24) and set the image filename
        # ComfyUI's LoadImage node looks for files in its 'input' directory
        workflow['24']['inputs']['image'] = "input_image.png"

    except Exception as e:
        print(f"Error loading or modifying workflow: {e}")
        return {'error': f'Failed to prepare workflow: {str(e)}'}

    # --- Run the Workflow ---
    try:
        generated_image_paths = runner.run_workflow(workflow)

        if not generated_image_paths:
            return {'error': 'Image generation failed, no output from ComfyUI.'}

        # --- Process and Return Output ---
        output_image_path = generated_image_paths[0]
        
        if not os.path.exists(output_image_path):
            return {'error': f'Generated image file not found at path: {output_image_path}'}
            
        # Read the generated image and encode it as base64
        with open(output_image_path, 'rb') as img_file:
            output_b64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Clean up the generated file
        os.remove(output_image_path)

        return {'image': output_b64}

    except Exception as e:
        print(f"An error occurred during workflow execution: {e}")
        return {'error': f'An unexpected error occurred: {str(e)}'}

# Start the serverless worker
if __name__ == "__main__":
    print("Starting Runpod serverless worker...")
    runpod.serverless.start({"handler": handler})
