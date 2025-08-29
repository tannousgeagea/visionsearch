
import os
import gc
import time
import torch
from cachetools import TTLCache
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
from common_utils.generative_ai.vlm.base import AnalysisType, VLMConfig, VLMResponse, ModelSource
from datetime import datetime, timedelta
import logging
import threading

import io

from PIL import Image, ImageDraw, ImageFont


def yolo_to_xyxy(bbox, image_width=1.0, image_height=1.0):
        """
        Converts a YOLO format bounding box (x_center, y_center, width, height)
        into (x_min, y_min, x_max, y_max) format.

        Parameters:
            bbox (list or tuple): [x_center, y_center, width, height] - all normalized (0 to 1)
            image_width (float): width of the image (used if you want pixel coordinates)
            image_height (float): height of the image (used if you want pixel coordinates)

        Returns:
            list: [x_min, y_min, x_max, y_max] - in normalized or pixel format
        """
        x_center, y_center, w, h = [float(el) for el in bbox]

        x_min = (x_center - w / 2) * image_width
        y_min = (y_center - h / 2) * image_height
        x_max = (x_center + w / 2) * image_width
        y_max = (y_center + h / 2) * image_height

        return [x_min, y_min, x_max, y_max]


# SOLUTION 1: Visual Bounding Box Overlay
def create_visual_bbox_image(image: Image.Image, bbox: list, 
                        highlight_color: str = "red", 
                        line_width: int = 5) -> Image.Image:
    """
    Create image with visible bounding box overlay
    This helps the model visually identify the region of interest
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Convert normalized coordinates to pixel coordinates
    width, height = image.size
    x_min = int(bbox[0] * width)
    y_min = int(bbox[1] * height)
    x_max = int(bbox[2] * width)
    y_max = int(bbox[3] * height)

    # Draw bounding box
    draw.rectangle(
        [(x_min, y_min), (x_max, y_max)], 
        outline=highlight_color, 
        width=line_width
    )

    # Add corner markers for better visibility
    corner_size = 10
    corners = [
        (x_min, y_min), (x_max, y_min), 
        (x_min, y_max), (x_max, y_max)
    ]

    for corner in corners:
        draw.rectangle([
            (corner[0] - corner_size, corner[1] - corner_size),
            (corner[0] + corner_size, corner[1] + corner_size)
        ], fill=highlight_color)

    # Add label
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw.text((x_min, y_min - 25), "TARGET OBJECT", 
            fill=highlight_color, font=font)


    img_copy.save('test_visual.png')
    return img_copy

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
    "/autolabel", methods=['POST']
)
async def autolabel(
    file: UploadFile = File(...),
    classes: List[str] = Form(...),
    xywhn: List[str] = Form(...)
):
    try:
        image_data = await file.read()
        config = config = VLMConfig(
            api_key="xddxdxddxdd",
            model_name="google/gemma-3-4b-it",
            model_source=ModelSource.OLLAMA,
            max_tokens=5000,
            enable_json=False
        )
        vlm = ModelTypeFactory.create_vlm(provider="gemma", config=config)

        system_prompt = (
            "You are a visual language model that analyzes waste environment images. "
            "Focus only on the objects highlighted with red boxes, describing them accurately "
            "without speculating about unmarked areas. "
            "For reference, a mattress is a rectangular object designed for sleeping or lying on, "
            "typically made of foam, springs, or other cushioning materials. "
            "Mattresses vary in size (single, double, queen, king) and can have different colors, patterns, or coverings."
        )


        user_prompt = (
            "Look at the objects in the red boxes and assign the most likely class name "
            f"from the provided list: {classes}. "
            "If uncertain or no match applies, return 'Unknown'. "
            "Return only the class label, without any extra text or explanation."
        )

        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        logging.error(xywhn)

        xyxy = yolo_to_xyxy(xywhn)
        create_visual_bbox_image(pil_image, xyxy)

        with open('test_visual.png', 'rb') as f:
            image_data = f.read()

        vlm.system_prompt = system_prompt
        response = vlm.analyze_image(
            image_data=image_data,
            prompt=user_prompt,
        )

       

        if not response.success:
            return HTTPException(
                status_code=500,
                detail=response.error_message
            )
        

        processing_time_ms = response.processing_time_ms
        processing_time_ms = 0

        model_info = "gemma3:12b-it-qat" 
        response_text = response.response_text

        return JSONResponse(
            status_code=200,
            content={
                "processing_time_ms": processing_time_ms,
                "max_memory_usage": torch.cuda.max_memory_allocated() / (1024 * 1024),
                "reserved_memory": torch.cuda.max_memory_reserved() / (1024 * 1024),
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
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()