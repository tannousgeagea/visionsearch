#!/usr/bin/env python3
"""
Ollama FastAPI Service

A comprehensive REST API wrapper for the Ollama adapter that allows external services
to interact with Ollama models through HTTP endpoints.

Features:
- Complete REST API for all Ollama functionality
- File upload support for images
- Streaming and non-streaming responses
- Proper error handling and validation
- API documentation with Swagger/OpenAPI
- Authentication support
- CORS configuration
- Health checks
"""


import json
import base64
import tempfile
import os
import time
import asyncio
import logging
import uvicorn
from pathlib import Path
from datetime import datetime
from fastapi.routing import APIRoute
from fastapi import Request, Response, APIRouter
from typing import List, Optional, Dict, Any, Union, Callable
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Import our Ollama adapter (assuming it's in the same directory or installed)
from common_utils.adapters.ollama.core import OllamaAdapter, ModelOptions, ChatMessage, Tool, OllamaAPIError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security (optional authentication)
security = HTTPBearer(auto_error=False)

# Global Ollama adapter instance
ollama_adapter = None

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
API_KEY = os.getenv("API_KEY")  # Optional API key for authentication
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size

class TimedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            print(f"route duration: {duration}")
            print(f"route response: {response}")
            print(f"route response headers: {response.headers}")
            return response

        return custom_route_handler


router = APIRouter(
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)

# ===================
# PYDANTIC MODELS
# ===================

class ModelOptionsRequest(BaseModel):
    """Request model for advanced model options"""
    num_keep: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    temperature: Optional[float] = Field(None, ge=0.0)
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mirostat: Optional[int] = Field(None, ge=0, le=2)
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

    def to_model_options(self) -> ModelOptions:
        """Convert to ModelOptions dataclass"""
        return ModelOptions(**self.dict(exclude_none=True))


class ChatMessageRequest(BaseModel):
    """Request model for chat messages"""
    role: str = Field(..., pattern="^(system|user|assistant|tool)$")
    content: str
    images: Optional[List[str]] = Field(None, description="Base64 encoded images")
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_chat_message(self) -> ChatMessage:
        """Convert to ChatMessage dataclass"""
        return ChatMessage(
            role=self.role,
            content=self.content,
            images=self.images,
            tool_calls=self.tool_calls
        )


class ToolRequest(BaseModel):
    """Request model for tools"""
    type: str = "function"
    function: Dict[str, Any]

    def to_tool(self) -> Tool:
        """Convert to Tool dataclass"""
        return Tool(type=self.type, function=self.function)


class GenerateRequest(BaseModel):
    """Request model for text generation"""
    model: str = Field(..., description="Model name")
    prompt: str = Field(..., description="Input prompt")
    suffix: Optional[str] = Field(None, description="Text after model response")
    images: Optional[List[str]] = Field(None, description="Base64 encoded images")
    format: Optional[str] = Field(None, pattern="^json$", description="Response format")
    options: Optional[ModelOptionsRequest] = None
    system: Optional[str] = Field(None, description="System message")
    template: Optional[str] = Field(None, description="Prompt template")
    context: Optional[List[int]] = Field(None, description="Context from previous request")
    stream: bool = Field(False, description="Stream response")
    raw: bool = Field(False, description="Raw mode")
    keep_alive: Optional[str] = Field("5m", description="Keep model loaded duration")


class ChatRequest(BaseModel):
    """Request model for chat completion"""
    model: str = Field(..., description="Model name")
    messages: List[ChatMessageRequest] = Field(..., description="Chat messages")
    tools: Optional[List[ToolRequest]] = Field(None, description="Available tools")
    format: Optional[str] = Field(None, pattern="^json$", description="Response format")
    options: Optional[ModelOptionsRequest] = None
    stream: bool = Field(False, description="Stream response")
    keep_alive: Optional[str] = Field("5m", description="Keep model loaded duration")


class CreateModelRequest(BaseModel):
    """Request model for model creation"""
    name: str = Field(..., description="Model name")
    modelfile: Optional[str] = Field(None, description="Modelfile content")
    path: Optional[str] = Field(None, description="Path to Modelfile")
    quantize: Optional[str] = Field(None, description="Quantization type")
    stream: bool = Field(False, description="Stream response")


class CopyModelRequest(BaseModel):
    """Request model for model copying"""
    source: str = Field(..., description="Source model name")
    destination: str = Field(..., description="Destination model name")


class DeleteModelRequest(BaseModel):
    """Request model for model deletion"""
    model: str = Field(..., description="Model name to delete")


class PullModelRequest(BaseModel):
    """Request model for pulling models"""
    model: str = Field(..., description="Model name to pull")
    insecure: bool = Field(False, description="Allow insecure connections")
    stream: bool = Field(False, description="Stream response")


class PushModelRequest(BaseModel):
    """Request model for pushing models"""
    model: str = Field(..., description="Model name to push (namespace/model:tag)")
    insecure: bool = Field(False, description="Allow insecure connections")
    stream: bool = Field(False, description="Stream response")


class EmbeddingRequest(BaseModel):
    """Request model for embeddings"""
    model: str = Field(..., description="Model name")
    input: Union[str, List[str]] = Field(..., description="Text or list of texts")
    truncate: bool = Field(True, description="Truncate to fit context")
    options: Optional[ModelOptionsRequest] = None
    keep_alive: Optional[str] = Field("5m", description="Keep model loaded duration")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    ollama_available: bool
    version: str = "1.0.0"


# ===================
# AUTHENTICATION
# ===================

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token if authentication is enabled"""
    if API_KEY is None:
        return True  # No authentication required
    
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    
    return True

# ===================
# STARTUP/SHUTDOWN
# ===================

@router.on_event("startup")
async def startup_event():
    """Initialize Ollama adapter on startup"""
    global ollama_adapter
    try:
        ollama_adapter = OllamaAdapter(base_url=OLLAMA_BASE_URL)
        logger.info(f"Initialized Ollama adapter with base URL: {OLLAMA_BASE_URL}")
        
        # Check if Ollama server is available
        if ollama_adapter.is_server_available():
            logger.info("Ollama server is available")
        else:
            logger.warning("Ollama server is not available - some endpoints will fail")
            
    except Exception as e:
        logger.error(f"Failed to initialize Ollama adapter: {e}")


@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Ollama API service")


# ===================
# UTILITY FUNCTIONS
# ===================

def handle_ollama_error(e: Exception):
    """Convert Ollama errors to HTTP exceptions"""
    if isinstance(e, OllamaAPIError):
        raise HTTPException(status_code=500, detail=str(e))
    else:
        # logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def stream_generator(generator):
    """Convert sync generator to async for streaming responses"""
    for item in generator:
        yield f"data: {json.dumps(item)}\n\n"


def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return base64 encoded content"""
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes"
        )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = file.file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Encode to base64
        encoded = base64.b64encode(content).decode('utf-8')
        return encoded
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


# ===================
# HEALTH CHECK
# ===================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    ollama_available = False
    if ollama_adapter:
        try:
            ollama_available = ollama_adapter.is_server_available()
        except:
            pass
    
    return HealthResponse(
        status="healthy" if ollama_available else "degraded",
        timestamp=datetime.now(),
        ollama_available=ollama_available
    )

# ===================
# GENERATION ENDPOINTS
# ===================

@router.post("/generate")
async def generate_text(
    request: GenerateRequest,
    _: bool = Depends(verify_token)
):
    """Generate text completion"""
    try:
        options = request.options.to_model_options() if request.options else None
        
        result = ollama_adapter.generate(
            model=request.model,
            prompt=request.prompt,
            suffix=request.suffix,
            images=request.images,
            format=request.format,
            options=options,
            system=request.system,
            template=request.template,
            context=request.context,
            stream=request.stream,
            raw=request.raw,
            keep_alive=request.keep_alive
        )
        
        if request.stream:
            return StreamingResponse(
                stream_generator(result),
                media_type="text/plain"
            )
        else:
            return result
            
    except Exception as e:
        handle_ollama_error(e)


@router.post("/generate/upload")
async def generate_with_images(
    model: str,
    prompt: str,
    files: List[UploadFile] = File(...),
    suffix: Optional[str] = None,
    format: Optional[str] = None,
    system: Optional[str] = None,
    stream: bool = False,
    _: bool = Depends(verify_token)
):
    """Generate text with uploaded image files"""
    try:
        # Process uploaded images
        images = []
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not an image"
                )
            encoded_image = save_uploaded_file(file)
            images.append(encoded_image)
        

        print(len(images))
        result = ollama_adapter.generate(
            model=model,
            prompt=prompt,
            suffix=suffix,
            images=images,
            format=format,
            system=system,
            stream=stream
        )
        
        if stream:
            return StreamingResponse(
                stream_generator(result),
                media_type="text/plain"
            )
        else:
            return result
            
    except Exception as e:
        handle_ollama_error(e)


@router.post("/chat")
async def chat_completion(
    request: ChatRequest,
    _: bool = Depends(verify_token)
):
    """Generate chat completion"""
    try:
        messages = [msg.to_chat_message() for msg in request.messages]
        tools = [tool.to_tool() for tool in request.tools] if request.tools else None
        options = request.options.to_model_options() if request.options else None
        
        result = ollama_adapter.chat(
            model=request.model,
            messages=messages,
            tools=tools,
            format=request.format,
            options=options,
            stream=request.stream,
            keep_alive=request.keep_alive
        )
        
        if request.stream:
            return StreamingResponse(
                stream_generator(result),
                media_type="text/plain"
            )
        else:
            return result
            
    except Exception as e:
        handle_ollama_error(e)


# ===================
# MODEL MANAGEMENT
# ===================

@router.get("/models")
async def list_models(_: bool = Depends(verify_token)):
    """List available models"""
    try:
        return ollama_adapter.list_models()
    except Exception as e:
        handle_ollama_error(e)


@router.get("/models/running")
async def list_running_models(_: bool = Depends(verify_token)):
    """List currently running models"""
    try:
        return ollama_adapter.list_running_models()
    except Exception as e:
        handle_ollama_error(e)


@router.post("/models/show")
async def show_model(
    model: str,
    verbose: bool = False,
    _: bool = Depends(verify_token)
):
    """Show model information"""
    try:
        return ollama_adapter.show_model(model, verbose)
    except Exception as e:
        handle_ollama_error(e)


@router.post("/models/create")
async def create_model(
    request: CreateModelRequest,
    _: bool = Depends(verify_token)
):
    """Create a new model"""
    try:
        result = ollama_adapter.create_model(
            name=request.name,
            modelfile=request.modelfile,
            path=request.path,
            quantize=request.quantize,
            stream=request.stream
        )
        
        if request.stream:
            return StreamingResponse(
                stream_generator(result),
                media_type="text/plain"
            )
        else:
            return result
            
    except Exception as e:
        handle_ollama_error(e)


@router.post("/models/copy")
async def copy_model(
    request: CopyModelRequest,
    _: bool = Depends(verify_token)
):
    """Copy a model"""
    try:
        ollama_adapter.copy_model(request.source, request.destination)
        return {"status": "success", "message": f"Model {request.source} copied to {request.destination}"}
    except Exception as e:
        handle_ollama_error(e)


@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    _: bool = Depends(verify_token)
):
    """Delete a model"""
    try:
        ollama_adapter.delete_model(model_name)
        return {"status": "success", "message": f"Model {model_name} deleted"}
    except Exception as e:
        handle_ollama_error(e)


@router.post("/models/pull")
async def pull_model(
    request: PullModelRequest,
    _: bool = Depends(verify_token)
):
    """Pull a model from registry"""
    try:
        result = ollama_adapter.pull_model(
            name=request.model,
            insecure=request.insecure,
            stream=request.stream
        )
        
        if request.stream:
            return StreamingResponse(
                stream_generator(result),
                media_type="text/plain"
            )
        else:
            return result
            
    except Exception as e:
        handle_ollama_error(e)


@router.post("/models/push")
async def push_model(
    request: PushModelRequest,
    _: bool = Depends(verify_token)
):
    """Push a model to registry"""
    try:
        result = ollama_adapter.push_model(
            name=request.model,
            insecure=request.insecure,
            stream=request.stream
        )
        
        if request.stream:
            return StreamingResponse(
                stream_generator(result),
                media_type="text/plain"
            )
        else:
            return result
            
    except Exception as e:
        handle_ollama_error(e)


# ===================
# EMBEDDING ENDPOINTS
# ===================

@router.post("/embeddings")
async def generate_embeddings(
    request: EmbeddingRequest,
    _: bool = Depends(verify_token)
):
    """Generate embeddings"""
    try:
        options = request.options.to_model_options() if request.options else None
        
        return ollama_adapter.generate_embeddings(
            model=request.model,
            input=request.input,
            truncate=request.truncate,
            options=options,
            keep_alive=request.keep_alive
        )
    except Exception as e:
        handle_ollama_error(e)


# ===================
# UTILITY ENDPOINTS
# ===================

@router.post("/models/load")
async def load_model(
    model: str,
    _: bool = Depends(verify_token)
):
    """Load a model into memory"""
    try:
        result = ollama_adapter.load_model(model)
        return {"status": "success", "message": f"Model {model} loaded", "details": result}
    except Exception as e:
        handle_ollama_error(e)


@router.post("/models/unload")
async def unload_model(
    model: str,
    _: bool = Depends(verify_token)
):
    """Unload a model from memory"""
    try:
        result = ollama_adapter.unload_model(model)
        return {"status": "success", "message": f"Model {model} unloaded", "details": result}
    except Exception as e:
        handle_ollama_error(e)


# ===================
# INFORMATION ENDPOINTS
# ===================

@router.get("/info")
async def get_api_info():
    """Get API information"""
    return {
        "name": "Ollama API Service",
        "version": "1.0.0",
        "ollama_base_url": OLLAMA_BASE_URL,
        "authentication_enabled": API_KEY is not None,
        "max_file_size": MAX_FILE_SIZE,
        "endpoints": {
            "generation": ["/api/generate", "/api/chat"],
            "models": ["/api/models", "/api/models/pull", "/api/models/create"],
            "embeddings": ["/api/embeddings"],
            "utilities": ["/api/models/load", "/api/models/unload"],
            "health": ["/health"]
        }
    }