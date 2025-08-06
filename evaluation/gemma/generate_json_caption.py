from glob import glob
import os

import cv2

import pipelines.gemma_image_objects_pipeline as gemma_image_objects_pipeline
import pipelines.gemma_image_object_iterative_pipeline as gemma_image_object_iterative_pipeline
import pipelines.gemma_batched_image_objects_pipeline as gemma_batched_image_objects_pipeline
import pipelines.gemma_batched_image_iterative_pipeline as gemma_batched_image_iterative_pipeline
import pipelines.gemma_batched_image_with_reference_pipeline as gemma_batched_image_with_reference_pipeline
from pipelines.reference_img import reference_img


from common_utils.generative_ai.vlm.gemma.core import GemmaVLM
from common_utils.generative_ai.vlm.base import VLMConfig

if __name__ == "__main__":
    imgs = glob("/home/appuser/src/bunker_imgs3/*")
    
    image_data = []

    for path2img in imgs:
        with open(path2img, "rb") as f:
                image_data.append(f.read())


    config = VLMConfig(
        api_key="xddxdxddxdd",
        model_name="google/gemma-3-4b-it",
        max_tokens=8000,
        enable_json=False
    )
    vlm = GemmaVLM(config=config, system_prompt= "")


    objects = ["pipes", "mattresses", "furniture", "metal objects"]

    gemma_image_objects_pipeline.pipeline(vlm=vlm, imgs=imgs, image_data=image_data, objects=objects, path2json="/home/appuser/src/captions/single_image_obj_list.json")
    gemma_image_object_iterative_pipeline.pipeline(vlm=vlm, imgs=imgs, image_data=image_data, objects=objects, path2json="/home/appuser/src/captions/single_image_obj_iterative.json")
    gemma_batched_image_objects_pipeline.pipeline(vlm=vlm, imgs=imgs, image_data=image_data, batch_size=4, objects=objects, path2json="/home/appuser/src/captions/batched_images_obj_list.json" )
    gemma_batched_image_iterative_pipeline.pipeline(vlm=vlm, imgs=imgs, image_data=image_data, batch_size=4, objects=objects, path2json="/home/appuser/src/captions/batched_images_obj_iterative.json")
    gemma_batched_image_with_reference_pipeline.pipeline(vlm=vlm,
                                                        imgs=imgs, 
                                                        image_data=image_data, 
                                                        batch_size=1, 
                                                        reference_dict=reference_img, 
                                                        path2json="/home/appuser/src/captions/image_with_cable_reference.json")