from common_utils.indexing.types import ImageData
from images.models import ImageAsset

def get_unindexed_images() -> list[ImageData]:
    return [
        ImageData(id=img.id, file_path=img.file_path)
        for img in ImageAsset.objects.filter(embedding_index__isnull=True)
    ]

def update_embedding_indices(indexed_ids: list[int]):
    for idx, image_id in enumerate(indexed_ids):
        ImageAsset.objects.filter(id=image_id).update(embedding_index=idx)
