
import os
import gc
import time
import torch
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from typing import List, Any
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File, Form
from common_utils.generative_ai.vlm import ModelTypeFactory
from common_utils.generative_ai.vlm.base import AnalysisType, VLMConfig, VLMResponse

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


vlm_models_cache = {}

def get_or_load_vlm(provider: str, model_name: str) -> Any:
    key = f"{provider}:{model_name}"
    if key not in vlm_models_cache:
        config = VLMConfig(
            api_key=os.getenv('GOOGLE_API_KEY', ''),
            model_name=model_name,
        )
        vlm_models_cache[key] = ModelTypeFactory.create_vlm(provider=provider, config=config)
    return vlm_models_cache[key]


router = APIRouter(
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)

@router.api_route(
    "/analyze-image", methods=['POST']
)
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    provider: str = Form(...),
    model_name: str = Form(...),
    analysis_type: AnalysisType = Form(default=AnalysisType.SCENE_UNDERSTANDING),
    prompt: Optional[str] = Form(default=None),
    detail_level: Optional[str] = Form(default="medium")
):
    try:
        image_data = await file.read()
        vlm = get_or_load_vlm(provider, model_name)

        # Use provided prompt if available, otherwise fall back to describe_scene
        if prompt:
            response: VLMResponse = vlm.analyze_image(
                image_data=image_data,
                prompt=prompt,
                analysis_type=analysis_type
            )
        else:
            response: VLMResponse = vlm.describe_scene(
                image_data=image_data,
                detail_level=detail_level or 'medium'
            )

        if not response.success:
            return HTTPException(
                status_code=500,
                detail=response.error_message
            )
        

        processing_time_ms = response.processing_time_ms
        analysis_type_value = response.analysis_type.value
        model_info = response.model_info
        response_text = response.response_text

        return JSONResponse(
            status_code=200,
            content={
                "processing_time_ms": processing_time_ms,
                "max_memory_usage": torch.cuda.max_memory_allocated() / (1024 * 1024),
                "reserved_memory": torch.cuda.max_memory_reserved() / (1024 * 1024),
                "analysis_type": analysis_type_value,
                # "detected_objects": response.detected_objects,
                # "extracted_text": response.extracted_text,
                "model_info": model_info,
                "response_text": response_text,
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        del image_data
        del response
        del prompt
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()