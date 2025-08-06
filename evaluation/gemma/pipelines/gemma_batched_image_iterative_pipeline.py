import time
import json
import re
import os

from pipelines.prompt import *

import re


def pipeline(vlm, imgs, image_data, batch_size,  objects, path2json):
    """
    USE BATCH FOR IMGS ON A SET OF OBJECTS
    """

    result_json = {}
    image_data_batched = []

    for x in range(0, round(len(image_data)) + 1, batch_size):
        image_data_batched.append(image_data[x:x + batch_size])

    
    for x, image_bytes in enumerate(image_data_batched):
        batch_key = f"batch_{x+1}"
        result_json[batch_key] = {} 
        img_keys = []
        for y in range(batch_size):
            if x*batch_size+y >  len(imgs) - 1:
                break
            img_key = os.path.basename(imgs[x*batch_size+y])
            img_keys.append(img_key)

            result_json[batch_key][img_key] = {}
            result_json[batch_key][img_key]["objects"] = []
    
        start_time = int(time.time())
        for obj in objects:
            prompt = get_object_user_prompt(obj, mode="batch")
            vlm.system_prompt = get_object_system_prompt(objects)
            vlm.config.enable_json = True

            response = vlm.analyze_multiple_images(
                images=image_bytes,
                prompt=prompt,
                json_obj_attributes={
                        "object_type": "",
                        "color": "<dominant_color_or_description>",
                        "material": "<inferred_material_if_possible>",
                        "location": "<'top left' | 'top' | 'top right' | 'center' | 'bottom left' | 'bottom' | 'bottom right'>",
                        "visibility": "<'fully visible' | 'partially occluded' | 'heavily occluded'>",
                        "size": "<'small' | 'medium' | 'large'>",
                        "image_number": "<Number as Int>"
                    },
                json_img_attributes = {
                "images":
                [
                    {
                        "image_description": "<Description of image>",
                        "image_caption": "<Caption generated from observed objects>",
                        "image_number": "<Number as Int>"
                    }
                ]
                }      
            )
            if not response.success:
                print(response.error_message)
            else:
                response_dict = response.raw_response
                for obj in response_dict["objects"]:
                    try:
                        img_key = img_keys[int(obj['object'].get("image_number")) - 1]  
                        result_json[batch_key][img_key]["objects"].append(obj)
                    except Exception:
                        continue
                
                for img_captions in response_dict['image_attributes']["images"]:
                    try:
                        img_key = img_keys[int(img_captions.get("image_number")) - 1]         
                        result_json[batch_key][img_key]['image_attributes'] = img_captions
                    except Exception:
                        result_json[batch_key][img_key]['image_attributes'] = {}
                                            
        result_json[batch_key]["processing_time"] =f"{time.time() - start_time:.2f}"
          
        vlm.config.enable_json = False
        vlm.system_prompt = caption_system_prompt

        start_time = int(time.time())

        json_objects = []
        for img_key in img_keys:
            json_objects.append(result_json[batch_key][img_key]['objects'])

        response = vlm.analyze_multiple_images(
                images=image_bytes,
                prompt=get_caption_user_prompt_for_batches(img_keys, json_objects)
            )
        if 'inference_with_image' not in result_json[batch_key]:
            result_json[batch_key]['inference_with_image'] = {}        
        result_json[batch_key]['inference_with_image'].update({"caption": response.response_text})
        result_json[batch_key]['inference_with_image'].update({"processing_time": f"{time.time() - start_time:.2f}"})

        start_time = int(time.time())
        response = vlm.analyze_multiple_images(
                images=[],
                prompt=get_caption_user_prompt_for_batches(img_keys, json_objects)
            )
        if not response.success:
            print(response.error_message)
        
        if 'inference_without_image' not in result_json[batch_key]:
            result_json[batch_key]['inference_without_image'] = {}

        result_json[batch_key]['inference_without_image'].update({"caption": response.response_text})
        result_json[batch_key]['inference_without_image'].update({"processing_time": f"{time.time() - start_time:.2f}"})

        with open(path2json, "w") as f:
            json.dump(result_json, f, indent=4)

    return result_json