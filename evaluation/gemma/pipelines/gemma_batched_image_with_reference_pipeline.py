import time
import json
import re
import os

from pipelines.prompt import *

import re


def pipeline(vlm, imgs, image_data, batch_size, reference_dict: dict, path2json: str):
    """
    USE BATCH FOR IMGS ON A SET OF OBJECTS
    """

    result_json = {}
    image_data_batched = []

    reference_img, reference_description = list(reference_dict.items())[1]
    with open(reference_img, "rb") as f:
        reference_img_bytes = f.read()

    # for x in range(0, round(len(image_data)) + 1, batch_size):
    #     image_data_batched.append(image_data[x:x + batch_size])

    for x, image_bytes in enumerate(image_data):
        img_key = os.path.basename(imgs[x])
        result_json[img_key] = {}
        result_json[img_key]["objects"] = []
        # batch_key = f"batch_{x+1}"
        # result_json[batch_key] = {} 
        # img_keys = []

        # for y in range(batch_size):
        #     if x*batch_size+y >  len(imgs) - 1:
        #         break
        #     img_key = os.path.basename(imgs[x*batch_size+y])
        #     img_keys.append(img_key)

        #     result_json[batch_key][img_key] = {}
        #     result_json[batch_key][img_key]["objects"] = []

        prompt = (
            "You are given one reference image along with its description. Use this reference to examine all the other images provided. "
            "Your goal is to identify objects in these images that are visually or contextually similar to the object in the reference. "
            "Focus on the following attributes when determining similarity: object type, material, color, location within the image, size, and condition. "
            "Do not repeat or rephrase the reference description. Instead, describe only the similar objects you find in the new images, using clear visual and spatial language. "
            "If no similar objects are found, state that clearly. "
            "Do not return any structured output (e.g., JSON) for the reference image — only for the additional images where similar objects are detected."
        )

        vlm.system_prompt = (
            "You are an AI assistant specialized in analyzing images from waste bunkers. "
            "You are capable of detecting and describing objects, accurately identifying variations in size, color, material, and condition. "
            "You also have the ability to compare a reference image with other images in order to identify objects that are visually or contextually similar. "
            "Use this capability to analyze and interpret waste bunker scenes, and respond with structured JSON containing only relevant object detections."
        )

        vlm.config.enable_json = True

        response = vlm.analyze_images_with_reference_image(
            reference_image=reference_img_bytes,
            reference_desription=reference_description,
            images=[image_bytes],
            prompt=prompt,
            json_obj_attributes={
                    "object_type": "",
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
                    },
                    "image_number": "<Number as Int>"
                },
        )
        if not response.success:
            print(response.error_message)
        else:
            response_dict = response.raw_response

            for obj in response_dict["objects"]:
                obj_dict = obj.get("object")

                img_number = obj_dict.get("image_number") - 1
                obj_dict.update({"image_number": img_number})
                if img_number == 0:
                    continue
                # img_key = img_keys[int(img_number) - 1]  
                # result_json[batch_key][img_key]["objects"].append(obj)
                result_json[img_key]["objects"].append(obj)


            # for img_captions in response_dict['image_attributes']["images"]:
            #     img_key = img_keys[int(img_captions.get("image_number")) - 2]         
            #     result_json[batch_key][img_key]['image_attributes'] = img_captions
   
            # result_json[batch_key]["processing_time"] =f"{response.processing_time_ms / 1000:.2f}"
            result_json[img_key]["processing_time"] =f"{response.processing_time_ms / 1000:.2f}"
        vlm.config.enable_json = False
        vlm.system_prompt = caption_system_prompt

        start_time = int(time.time())

        json_objects = result_json[img_key]['objects']
        print(get_caption_user_prompt(json_objects))
        # for img_key in img_keys:
        #     json_objects.append(result_json[batch_key][img_key]['objects'])

        response = vlm.analyze_image(
                image_data=image_bytes,
                prompt=get_caption_user_prompt(json_objects)
            )
        # if 'inference_with_image' not in result_json[batch_key]:
        #     result_json[batch_key]['inference_with_image'] = {}        
        # result_json[batch_key]['inference_with_image'].update({"caption": response.response_text})
        # result_json[batch_key]['inference_with_image'].update({"processing_time": f"{time.time() - start_time:.2f}"})

        if 'inference_with_image' not in result_json[img_key]:
            result_json[img_key]['inference_with_image'] = {}        
        result_json[img_key]['inference_with_image'].update({"caption": response.response_text})
        result_json[img_key]['inference_with_image'].update({"processing_time": f"{time.time() - start_time:.2f}"})

        start_time = int(time.time())
        response = vlm.analyze_image(
                image_data=[],
                prompt=get_caption_user_prompt(json_objects)
            )
        if not response.success:
            print(response.error_message)
        
        # if 'inference_without_image' not in result_json[batch_key]:
        # result_json[batch_key]['inference_without_image'] = {}
        if 'inference_without_image' not in result_json[img_key]:
            result_json[img_key]['inference_without_image'] = {}

        # result_json[batch_key]['inference_without_image'].update({"caption": response.response_text})
        # result_json[batch_key]['inference_without_image'].update({"processing_time": f"{time.time() - start_time:.2f}"})

        result_json[img_key]['inference_without_image'].update({"caption": response.response_text})
        result_json[img_key]['inference_without_image'].update({"processing_time": f"{time.time() - start_time:.2f}"})

        with open(path2json, "w") as f:
            json.dump(result_json, f, indent=4)

    return result_json