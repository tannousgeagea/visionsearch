
import os
import time
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from typing import List
from pydantic import BaseModel
from typing import AsyncIterator
import json
from fastapi.responses import StreamingResponse
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


router = APIRouter(
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)

@router.api_route(
    "/analyze-image/stream", methods=['POST']
)
async def stream_analyze_image(
    file: UploadFile = File(...),
    provider: str = File(...),
    model_name: str = Form(...),
    prompt: str = Form(...)
):
    try:
        image_data = await file.read()
        config = VLMConfig(
            api_key=os.getenv('GOOGLE_API_KEY', ''),
            model_name=model_name,
            enable_streaming=True
        )
        vlm = ModelTypeFactory.create_vlm(provider=provider, config=config)

        async def generate() -> AsyncIterator[bytes]:
            async for chunk in vlm.analyze_image_stream(image_data=image_data, prompt=prompt):
                json_chunk = json.dumps(chunk.__dict__) + "\n"
                yield json_chunk.encode("utf-8")

        return StreamingResponse(generate(), media_type="application/json")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))