#!/usr/bin/env python3
"""
Comprehensive Ollama API Adapter

This adapter provides a complete interface to all Ollama API endpoints,
covering all cases mentioned in the official documentation.

Features:
- Generate completions (streaming and non-streaming)
- Chat completions with conversation history
- Model management (create, list, show, copy, delete)
- Pull and push models
- Generate embeddings
- Multimodal support (images)
- Tool calling support
- All advanced parameters and options
"""

import json
import base64
import requests
from typing import Dict, List, Union, Optional, Generator, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time


@dataclass
class ModelOptions:
    """Advanced model parameters for fine-tuning generation"""
    num_keep: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    temperature: Optional[float] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_newline: Optional[bool] = None
    stop: Optional[List[str]] = None
    numa: Optional[bool] = None
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = None
    num_gpu: Optional[int] = None
    main_gpu: Optional[int] = None
    low_vram: Optional[bool] = None
    vocab_only: Optional[bool] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    num_thread: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # system, user, assistant, tool
    content: str
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests"""
        result = {"role": self.role, "content": self.content}
        if self.images:
            result["images"] = self.images
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result


@dataclass
class Tool:
    """Represents a function tool for model use"""
    type: str = "function"
    function: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "function": self.function}


class OllamaAPIError(Exception):
    """Custom exception for Ollama API errors"""
    pass


class OllamaAdapter:
    """Comprehensive Ollama API Adapter"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama adapter
        
        Args:
            base_url: Base URL for the Ollama server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make a request to the Ollama API"""
        url = f"{self.base_url}/api{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            if response.status_code >= 400:
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                raise OllamaAPIError(f"API request failed: {response.status_code} - {error_detail}")
            return response
        except requests.exceptions.RequestException as e:
            raise OllamaAPIError(f"Request failed: {str(e)}")

    def _stream_response(self, response: requests.Response) -> Generator[Dict[str, Any], None, None]:
        """Handle streaming responses"""
        for line in response.iter_lines():
            if line:
                try:
                    yield json.loads(line.decode('utf-8'))
                except json.JSONDecodeError as e:
                    raise OllamaAPIError(f"Failed to parse streaming response: {str(e)}")

    # ====================
    # GENERATION ENDPOINTS
    # ====================

    def generate(
        self,
        model: str,
        prompt: str,
        suffix: Optional[str] = None,
        images: Optional[List[Union[str, Path]]] = None,
        format: Optional[str] = None,
        options: Optional[ModelOptions] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = True,
        raw: bool = False,
        keep_alive: Optional[str] = None
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Generate a completion for the given prompt
        
        Args:
            model: Model name (required)
            prompt: The prompt to generate a response for
            suffix: Text after the model response (for code completion)
            images: List of base64-encoded images or file paths
            format: Response format ("json" for JSON mode)
            options: Advanced model parameters
            system: System message (overrides Modelfile)
            template: Prompt template (overrides Modelfile)  
            context: Context from previous request for conversation
            stream: Whether to stream the response
            raw: Whether to use raw mode (bypass templating)
            keep_alive: How long to keep model loaded (default: "5m")
            
        Returns:
            Generator of response objects if streaming, single response object if not
        """
        # Process images
        processed_images = None
        if images:
            processed_images = []
            for img in images:
                if isinstance(img, (str, Path)):
                    if Path(img).exists():
                        # Read from file and encode
                        with open(img, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                        processed_images.append(img_data)
                    else:
                        # Assume it's already base64 encoded
                        processed_images.append(str(img))

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

        # Add optional parameters
        if suffix:
            payload["suffix"] = suffix
        if processed_images:
            payload["images"] = processed_images
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options.to_dict()
        if system:
            payload["system"] = system
        if template:
            payload["template"] = template
        if context:
            payload["context"] = context
        if raw:
            payload["raw"] = raw
        if keep_alive:
            payload["keep_alive"] = keep_alive

        response = self._request("POST", "/generate", json=payload, stream=stream)
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()

    def chat(
        self,
        model: str,
        messages: List[Union[ChatMessage, Dict[str, Any]]],
        tools: Optional[List[Union[Tool, Dict[str, Any]]]] = None,
        format: Optional[str] = None,
        options: Optional[ModelOptions] = None,
        stream: bool = True,
        keep_alive: Optional[str] = None
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Generate a chat completion
        
        Args:
            model: Model name (required)
            messages: List of chat messages
            tools: Tools for the model to use (requires stream=False)
            format: Response format ("json" for JSON mode)
            options: Advanced model parameters
            stream: Whether to stream the response
            keep_alive: How long to keep model loaded
            
        Returns:
            Generator of response objects if streaming, single response object if not
        """
        # Process messages
        processed_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                processed_messages.append(msg.to_dict())
            else:
                processed_messages.append(msg)

        # Process tools
        processed_tools = None
        if tools:
            processed_tools = []
            for tool in tools:
                if isinstance(tool, Tool):
                    processed_tools.append(tool.to_dict())
                else:
                    processed_tools.append(tool)
            # Tools require stream=False
            stream = False

        payload = {
            "model": model,
            "messages": processed_messages,
            "stream": stream
        }

        # Add optional parameters
        if processed_tools:
            payload["tools"] = processed_tools
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options.to_dict()
        if keep_alive:
            payload["keep_alive"] = keep_alive

        response = self._request("POST", "/chat", json=payload, stream=stream)
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()

    # ==================
    # MODEL MANAGEMENT
    # ==================

    def create_model(
        self,
        name: str,
        modelfile: Optional[str] = None,
        path: Optional[str] = None,
        quantize: Optional[str] = None,
        stream: bool = True
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Create a model from a Modelfile
        
        Args:
            name: Name of the model to create
            modelfile: Contents of the Modelfile
            path: Path to the Modelfile
            quantize: Quantization type (e.g., "q4_K_M", "q8_0")
            stream: Whether to stream the response
            
        Returns:
            Generator of status objects if streaming, single response if not
        """
        payload = {
            "model": name,
            "stream": stream
        }

        if modelfile:
            payload["modelfile"] = modelfile
        if path:
            payload["path"] = path
        if quantize:
            payload["quantize"] = quantize

        response = self._request("POST", "/create", json=payload, stream=stream)
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()

    def list_models(self) -> Dict[str, Any]:
        """List models available locally"""
        response = self._request("GET", "/tags")
        return response.json()

    def show_model(self, name: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Show information about a model
        
        Args:
            name: Model name
            verbose: Include full data for verbose fields
            
        Returns:
            Model information
        """
        payload = {"model": name}
        if verbose:
            payload["verbose"] = verbose

        response = self._request("POST", "/show", json=payload)
        return response.json()

    def copy_model(self, source: str, destination: str) -> None:
        """
        Copy a model
        
        Args:
            source: Source model name
            destination: Destination model name
        """
        payload = {
            "source": source,
            "destination": destination
        }
        self._request("POST", "/copy", json=payload)

    def delete_model(self, name: str) -> None:
        """
        Delete a model
        
        Args:
            name: Model name to delete
        """
        payload = {"model": name}
        self._request("DELETE", "/delete", json=payload)

    def pull_model(
        self,
        name: str,
        insecure: bool = False,
        stream: bool = True
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Pull a model from the library
        
        Args:
            name: Model name to pull
            insecure: Allow insecure connections
            stream: Whether to stream the response
            
        Returns:
            Generator of download progress if streaming, single response if not
        """
        payload = {
            "model": name,
            "stream": stream
        }
        if insecure:
            payload["insecure"] = insecure

        response = self._request("POST", "/pull", json=payload, stream=stream)
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()

    def push_model(
        self,
        name: str,
        insecure: bool = False,
        stream: bool = True
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Push a model to the library
        
        Args:
            name: Model name to push (format: namespace/model:tag)
            insecure: Allow insecure connections
            stream: Whether to stream the response
            
        Returns:
            Generator of upload progress if streaming, single response if not
        """
        payload = {
            "model": name,
            "stream": stream
        }
        if insecure:
            payload["insecure"] = insecure

        response = self._request("POST", "/push", json=payload, stream=stream)
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()

    def list_running_models(self) -> Dict[str, Any]:
        """List models currently loaded into memory"""
        response = self._request("GET", "/ps")
        return response.json()

    # ==================
    # EMBEDDING ENDPOINTS
    # ==================

    def generate_embeddings(
        self,
        model: str,
        input: Union[str, List[str]],
        truncate: bool = True,
        options: Optional[ModelOptions] = None,
        keep_alive: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings from a model
        
        Args:
            model: Model name
            input: Text or list of text to generate embeddings for
            truncate: Truncate input to fit context length
            options: Additional model parameters
            keep_alive: How long to keep model loaded
            
        Returns:
            Embeddings response
        """
        payload = {
            "model": model,
            "input": input,
            "truncate": truncate
        }

        if options:
            payload["options"] = options.to_dict()
        if keep_alive:
            payload["keep_alive"] = keep_alive

        response = self._request("POST", "/embed", json=payload)
        return response.json()

    def generate_embedding_legacy(
        self,
        model: str,
        prompt: str,
        options: Optional[ModelOptions] = None,
        keep_alive: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate embedding (legacy endpoint)
        Note: This endpoint has been superseded by /api/embed
        
        Args:
            model: Model name
            prompt: Text to generate embeddings for
            options: Additional model parameters
            keep_alive: How long to keep model loaded
            
        Returns:
            Embedding response
        """
        payload = {
            "model": model,
            "prompt": prompt
        }

        if options:
            payload["options"] = options.to_dict()
        if keep_alive:
            payload["keep_alive"] = keep_alive

        response = self._request("POST", "/embeddings", json=payload)
        return response.json()

    # ==================
    # BLOB MANAGEMENT
    # ==================

    def check_blob_exists(self, digest: str) -> bool:
        """
        Check if a blob exists on the server
        
        Args:
            digest: SHA256 digest of the blob
            
        Returns:
            True if blob exists, False otherwise
        """
        try:
            response = self._request("HEAD", f"/blobs/{digest}")
            return response.status_code == 200
        except OllamaAPIError:
            return False

    def create_blob(self, digest: str, file_path: Union[str, Path]) -> None:
        """
        Create a blob from a file
        
        Args:
            digest: Expected SHA256 digest of the file
            file_path: Path to the file to upload
        """
        with open(file_path, 'rb') as f:
            self._request("POST", f"/blobs/{digest}", data=f)

    # ==================
    # UTILITY METHODS
    # ==================

    def load_model(self, model: str) -> Dict[str, Any]:
        """Load a model into memory (using empty prompt)"""
        return self.generate(model=model, prompt="", stream=False)

    def unload_model(self, model: str) -> Dict[str, Any]:
        """Unload a model from memory"""
        return self.generate(model=model, prompt="", keep_alive="0", stream=False)

    def load_model_chat(self, model: str) -> Dict[str, Any]:
        """Load a model into memory using chat endpoint"""
        return self.chat(model=model, messages=[], stream=False)

    def unload_model_chat(self, model: str) -> Dict[str, Any]:
        """Unload a model from memory using chat endpoint"""
        return self.chat(model=model, messages=[], keep_alive="0", stream=False)

    def encode_image_from_path(self, image_path: Union[str, Path]) -> str:
        """
        Encode an image file to base64
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def is_server_available(self) -> bool:
        """Check if the Ollama server is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def wait_for_server(self, timeout: int = 30, interval: int = 1) -> bool:
        """
        Wait for the server to become available
        
        Args:
            timeout: Maximum time to wait in seconds
            interval: Check interval in seconds
            
        Returns:
            True if server becomes available, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_available():
                return True
            time.sleep(interval)
        return False


# ====================
# EXAMPLE USAGE
# ====================

if __name__ == "__main__":
    # Initialize the adapter
    adapter = OllamaAdapter()
    
    # Check server availability
    if not adapter.is_server_available():
        print("Ollama server is not available. Please start Ollama first.")
        exit(1)
    
    try:
        # Example 1: List available models
        print("Available models:")
        models = adapter.list_models()
        for model in models["models"]:
            print(f"  - {model['name']}")
        
        if not models["models"]:
            print("No models found. Please pull a model first (e.g., ollama pull llama3.2)")
            exit(1)
        
        model_name = models["models"][0]["name"]
        print(f"\nUsing model: {model_name}")
        
        # Example 2: Simple generation (non-streaming)
        print("\n=== Simple Generation ===")
        response = adapter.generate(
            model=model_name,
            prompt="Why is the sky blue?",
            stream=False
        )
        print(f"Response: {response['response']}")
        
        # Example 3: Chat with conversation
        print("\n=== Chat Conversation ===")
        messages = [
            ChatMessage(role="user", content="Hello! What's your name?")
        ]
        
        response = adapter.chat(
            model=model_name,
            messages=messages,
            stream=False
        )
        print(f"Assistant: {response['message']['content']}")
        
        # Example 4: Streaming generation
        print("\n=== Streaming Generation ===")
        print("Response: ", end="", flush=True)
        for chunk in adapter.generate(
            model=model_name,
            prompt="Tell me a short joke",
            stream=True
        ):
            if not chunk.get('done', False):
                print(chunk.get('response', ''), end='', flush=True)
        print()  # New line after streaming
        
        # Example 5: Generation with advanced options
        print("\n=== Generation with Options ===")
        options = ModelOptions(
            temperature=0.8,
            top_p=0.9,
            seed=42
        )
        
        response = adapter.generate(
            model=model_name,
            prompt="Write a creative sentence about space",
            options=options,
            stream=False
        )
        print(f"Creative response: {response['response']}")
        
        print("\n=== All Examples Completed Successfully ===")
        
    except OllamaAPIError as e:
        print(f"Ollama API Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")