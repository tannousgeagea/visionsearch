import time
import json
import os

from pipelines.prompt import *

def pipeline(vlm, imgs, image_data, objects):
    """
    SINGLE IMAGE PROMPT CAPTION LIST OF OBJECTS
    """

    result_json = {}
    for x, image_bytes in enumerate(image_data):
        img_key = os.path.basename(imgs[x])
        prompt = get_object_user_prompt(objects)
        vlm.system_prompt = get_object_system_prompt(objects)
        vlm.config.enable_json = True

        response = vlm.analyze_multiple_images(
            images=image_bytes,
            prompt=prompt,
            json_obj_attributes={
                    "object_type": "<'pipe' | 'mattress' | 'furniture' | 'metal object'>",
                    "color": "<dominant_color_or_description>",
                    "material": "<inferred_material_if_possible>",
                    "location": "<'top left' | 'top' | 'top right' | 'center' | 'bottom left' | 'bottom' | 'bottom right'>",
                    "visibility": "<'fully visible' | 'partially occluded' | 'heavily occluded'>",
                    "size": "<'small' | 'medium' | 'large'>",
                    "bounding_box": {
                        "x_min": "<float between 0.0 and 1.0>",
                        "y_min": "<float between 0.0 and 1.0>",
                        "x_max": "<float between 0.0 and 1.0>",
                        "y_max":" <float between 0.0 and 1.0>"
                    }
                },
            json_img_attributes={
                "image_description": "<Description of image>",
                "image_caption": "<Caption generated from observed objects>"
                },
        )
        if not response.success:
            print(response.error_message)
        else:
            response_dict = response.raw_response
            result_json.update({img_key: {}})
            for (k,v) in response_dict.items():              
                result_json.get(img_key).update({k:v})
                result_json.get(img_key).update({"processing_time": f"{response.processing_time_ms / 1000:.2f}"})
          
        vlm.config.enable_json = False
        vlm.system_prompt = caption_system_prompt

        start_time = int(time.time())
        response = vlm.analyze_multiple_images(
                images=[image_bytes],
                prompt=get_caption_user_prompt(result_json.get(os.path.basename(imgs[x])).get("objects"))
            )
        if 'inference_with_image' not in result_json[img_key]:
            result_json[img_key]['inference_with_image'] = {}

        result_json[img_key]['inference_with_image'].update({"caption": response.response_text})
        result_json[img_key]['inference_with_image'].update({"processing_time": f"{time.time() - start_time:.2f}"})

        start_time = int(time.time())
        response = vlm.analyze_multiple_images(
                images=[],
                prompt=get_caption_user_prompt(result_json.get(img_key).get("objects"))
            )
        if not response.success:
            print(response.error_message)
        
        if 'inference_without_image' not in result_json[img_key]:
            result_json[img_key]['inference_without_image'] = {}

        result_json[img_key]['inference_without_image'].update({"caption": response.response_text})
        result_json[img_key]['inference_without_image'].update({"processing_time": f"{time.time() - start_time:.2f}"})

        with open("/home/appuser/src/captions/single_image_obj_list.json", "w") as f:
            json.dump(result_json, f, indent=4)

    return result_json