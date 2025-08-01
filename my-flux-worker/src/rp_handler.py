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

def find_node_and_set_widget_value(workflow, node_id, widget_index=0, value=None):
    """Helper function to find a node by ID and update a widget's value."""
    for node in workflow.get("nodes", []):
        if str(node.get("id")) == str(node_id):
            if "widgets_values" in node and len(node["widgets_values"]) > widget_index:
                node["widgets_values"][widget_index] = value
                print(f"Updated widget {widget_index} for node {node_id} to '{value}'")
                return True
    return False

def handler(job):
    """
    This is the main handler function that Runpod will call.
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
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data))
        
        input_dir = "/ComfyUI/input"
        os.makedirs(input_dir, exist_ok=True)
        input_image_path = os.path.join(input_dir, "input_image.png")
        
        image.save(input_image_path)
        print(f"Input image saved to: {input_image_path}")
        
    except Exception as e:
        print(f"Error processing input image: {e}")
        return {'error': f'Failed to process input image: {str(e)}'}

    # --- Load and Modify Workflow ---
    try:
        with open('/workflow_api.json', 'r') as f:
            workflow = json.load(f)

        # CORRECTED: Find the prompt node (ID 9) and set its text widget
        # The prompt is the first widget in the CLIPTextEncode node.
        if not find_node_and_set_widget_value(workflow, 9, 0, prompt):
            return {'error': 'Could not find prompt node (ID 9) in workflow.'}

        # CORRECTED: Find the LoadImage node (ID 24) and set the image filename
        # The filename is the first widget in the LoadImage node.
        if not find_node_and_set_widget_value(workflow, 24, 0, "input_image.png"):
            return {'error': 'Could not find Load Image node (ID 24) in workflow.'}

    except Exception as e:
        print(f"Error loading or modifying workflow: {e}")
        return {'error': f'Failed to prepare workflow: {str(e)}'}

    # --- Run the Workflow ---
    try:
        # NOTE: The workflow must be converted to the "prompt" format for the API call.
        # This creates a dictionary where keys are node IDs.
        prompt_workflow = {str(node["id"]): node for node in workflow["nodes"]}
        generated_image_paths = runner.run_workflow(prompt_workflow)

        if not generated_image_paths:
            return {'error': 'Image generation failed, no output from ComfyUI.'}

        # --- Process and Return Output ---
        output_image_path = generated_image_paths[0]
        
        if not os.path.exists(output_image_path):
            return {'error': f'Generated image file not found at path: {output_image_path}'}
            
        with open(output_image_path, 'rb') as img_file:
            output_b64 = base64.b64encode(img_file.read()).decode('utf-8')

        os.remove(output_image_path)
        return {'image': output_b64}

    except Exception as e:
        print(f"An error occurred during workflow execution: {e}")
        return {'error': f'An unexpected error occurred: {str(e)}'}

# Start the serverless worker
if __name__ == "__main__":
    print("Starting Runpod serverless worker...")
    runpod.serverless.start({"handler": handler})
