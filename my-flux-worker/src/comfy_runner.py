import subprocess
import threading
import time
import websocket
import uuid
import json
import os
import urllib.request
import urllib.parse

class ComfyRunner:
    """
    A class to manage running a ComfyUI server and executing workflows.
    """
    def __init__(self, server_address="127.0.0.1:8188"):
        """
        Initializes the ComfyRunner.

        Args:
            server_address (str): The address for the ComfyUI server.
        """
        self.server_address = server_address
        self.server_thread = None
        self.server_process = None
        self._start_server()

    def _start_server(self):
        """
        Starts the ComfyUI server in a separate thread.
        """
        print("Starting ComfyUI server...")
        self.server_thread = threading.Thread(target=self._run_server_process)
        self.server_thread.daemon = True
        self.server_thread.start()
        self._wait_for_server()

    def _run_server_process(self):
        """
        Executes the ComfyUI main.py script as a subprocess.
        """
        # Command to start the ComfyUI server
        command = ["python", "main.py", f"--listen={self.server_address.split(':')[0]}"]
        
        # Run the subprocess
        self.server_process = subprocess.Popen(
            command,
            cwd='/ComfyUI',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Optional: Log stdout and stderr for debugging
        for line in iter(self.server_process.stdout.readline, ''):
            print(f"[ComfyUI STDOUT] {line.strip()}")
        for line in iter(self.server_process.stderr.readline, ''):
            print(f"[ComfyUI STDERR] {line.strip()}")

    def _wait_for_server(self, timeout=60):
        """
        Waits for the ComfyUI server to become responsive.

        Args:
            timeout (int): Maximum time to wait in seconds.
        
        Raises:
            RuntimeError: If the server does not start within the timeout period.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Attempt to connect to the server's websocket
                ws_url = f"ws://{self.server_address}/ws?clientId={uuid.uuid4()}"
                ws = websocket.create_connection(ws_url, timeout=5)
                ws.close()
                print("ComfyUI server is ready.")
                return
            except (websocket.WebSocketException, ConnectionRefusedError, ConnectionResetError, TimeoutError):
                time.sleep(1)  # Wait and retry
        raise RuntimeError("ComfyUI server failed to start in time.")

    def _queue_prompt(self, prompt_workflow):
        """
        Queues a prompt (workflow) for execution in ComfyUI.

        Args:
            prompt_workflow (dict): The ComfyUI workflow to execute.

        Returns:
            list: A list of file paths for the generated images.
        """
        client_id = str(uuid.uuid4())
        ws_url = f"ws://{self.server_address}/ws?clientId={client_id}"
        
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        
        try:
            # Send the prompt to the server
            prompt_data = {"prompt": prompt_workflow, "client_id": client_id}
            ws.send(json.dumps(prompt_data))
            
            # Listen for messages from the server
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executed':
                        # This message type indicates the workflow is done
                        data = message['data']
                        output_images = data.get('output', {}).get('images', [])
                        
                        image_paths = []
                        for image_info in output_images:
                            # Construct the full path to the output image
                            filename = image_info['filename']
                            subfolder = image_info['subfolder']
                            full_path = os.path.join('/ComfyUI/output', subfolder, filename)
                            image_paths.append(full_path)
                        return image_paths
                else:
                    # Binary data (previews) can be ignored
                    continue
        finally:
            ws.close()
        return []

    def run_workflow(self, workflow):
        """
        A wrapper function to run a workflow.

        Args:
            workflow (dict): The ComfyUI workflow.

        Returns:
            list: A list of file paths for the generated images.
        """
        print("Queueing prompt with ComfyUI...")
        return self._queue_prompt(workflow)
