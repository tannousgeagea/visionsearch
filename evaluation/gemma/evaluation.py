from glob import glob

import pipelines.gemma_image_objects_pipeline as gemma_image_objects_pipeline
import pipelines.gemma_image_object_iterative_pipeline as gemma_image_object_iterative_pipeline

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
        max_tokens=5000,
        enable_json=True
    )
    vlm = GemmaVLM(config=config, system_prompt="")


    objects = [["pipes", "mattresses", "furniture", "metal objects"]]

    # gemma_image_objects_pipeline.pipeline(vlm=vlm, imgs=imgs, image_data=image_data, objects=objects)
    gemma_image_object_iterative_pipeline.pipeline(vlm=vlm, imgs=imgs, image_data=image_data, objects=objects, path2json="/home/appuser/src/captions/single_image_obj_iterative.json")