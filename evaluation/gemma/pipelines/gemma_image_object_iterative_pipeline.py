import time
import json
import os

from pipelines.prompt import *

def pipeline(vlm, imgs, image_data, objects):
    """
    SINGLE IMAGE PROMPT CAPTION SINGLE OBJ PER ITERATION
    """
    result_json = {}

    for x, image_bytes in enumerate(image_data):
        img_key = os.path.basename(imgs[x])
        result_json[img_key] = {"objects": []}
        
        start_time = int(time.time())

        for obj in objects:
            prompt = get_object_user_prompt(obj)

            
            vlm.system_prompt = get_object_system_prompt(obj)
            vlm.config.enable_json = True

            response = vlm.analyze_multiple_images(
                images=image_bytes,
                prompt=prompt,
                json_obj_attributes={
                        "object_type": f"<{obj}>",
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
                    },
            )
            if not response.success:
                print(response.error_message)
            else:
                response_dict = response.raw_response
                for (k,v) in response_dict.items():
                    if k == "objects":              
                        objects = result_json.get(img_key).get("objects")
                        objects.extend(v) 
            
        result_json.get(os.path.basename(imgs[x])).update({"processing_time": f"{time.time() - start_time:.2f}"})   

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

        with open("/home/appuser/src/captions/single_image_obj_iterative.json", "w") as f:
            json.dump(result_json, f, indent=4)
    
    return result_json