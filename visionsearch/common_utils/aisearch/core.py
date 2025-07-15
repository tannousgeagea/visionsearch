import os
import json
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List
from common_utils.indexing.types import ImageData
from common_utils.embedding import create as create_embedding
from common_utils.indexing import create as create_index
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class VisionAISearch:
    def __init__(self, index_path: str= "faiss.index", model_name:str="clip", backend:str="faiss"):
        self.index_path = index_path
        self.id_map_path = f"{index_path}.ids.npy"
        self.asset_ids: List[int] = []
        self.embedder = create_embedding(model_name)
        self.index = create_index(backend=backend, dim=self.embedder.get_embed_dim())
        self._load_if_exists()

    def _load_if_exists(self):
        if Path(self.index_path).exists() and Path(self.id_map_path).exists():
            self.index.load(self.index_path)
            self.asset_ids = np.load(self.id_map_path).tolist()

    def build(self, images:List[ImageData]):
        vectors = []
        ids= []
        logging.info("Building FAISS index from images...")

        pbar = tqdm(images, ncols=100)
        for img in pbar:
            if img.id in self.asset_ids:
                continue

            vector = self.embedder.extract_image_feature(image=img.file_path)
            vectors.append(vector)
            ids.append(img.id)
        
        if vectors:
            matrix = np.vstack(vectors).astype(np.float32)
            logging.info(matrix.shape)
            self.index.add(matrix)
            self.asset_ids.extend(ids)

        logging.info(f"Indexed {len(ids)} images.")

    def load(self):
        self.index.load(self.index_path)
        self.asset_ids = np.load(self.id_map_path).tolist()

    def save(self):
        self.index.save(self.index_path)
        np.save(self.id_map_path, np.array(self.asset_ids))

    def search(self, query: str, k=5, similiraty_threshold:float=0.1):
        text_feat = self.embedder.extract_text_feature(query).astype("float32")
        indices, distances = self.index.search(text_feat, k)
        results = [
            (self.asset_ids[i], float(distances[idx])) for idx, i in enumerate(indices) if distances[idx] >= similiraty_threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        print("\nRanked Results:")
        for name, score in results:
            print(f"  - {name} | Similarity: {score:.4f}")

        return results
    

class VisionAISearchWithCaption(VisionAISearch):
    def __init__(self, index_path: str= "faiss.index", model_name:str="perception", backend:str="faiss", path2caption: str=""):
        super().__init__(index_path, model_name, backend)
        self.id_map_path = f"{index_path}.ids.npy"
        if not os.path.exists(path2caption):
            raise FileNotFoundError(f"Caption file doesn't exist at path {path2caption}")

        with open(path2caption) as f:
            self.captions = json.load(f)

    def build(self, images:List[ImageData]):
        vectors = []
        ids= []
        logging.info("Building FAISS index from image captions...")

        indices = len(images)
        pbar = tqdm(range(indices), ncols=100)
        for i in pbar:
            img = images[i]
            if img.id in self.asset_ids or img.file_path not in self.captions.keys():
                continue
            
            caption = self.captions.get(img.file_path)
            temp_vectors = []
            for sentence in caption:
                vector = self.embedder.extract_text_feature(sentence)
                # temp_vectors.append(vector)
            
                # vectors.append(np.mean(temp_vectors, axis=1))
                vectors.append(vector)
                ids.append(img.id)
        
        # Todo: Multiple embeddings are created for one img, change that
        if vectors:
            matrix = np.vstack(vectors).astype(np.float32)
            logging.info(matrix.shape)
            self.index.add(matrix)
            self.asset_ids.extend(ids)

        logging.info(f"Indexed {len(ids)} images.")
        
if __name__ == "__main__":
    from glob import glob
    EMBEDDING_BACKEND = os.getenv("VISIONSEARCH_EMBEDDING_BACKEND", "clip")
    INDEX_BACKEND = os.getenv("VISIONSEARCH_INDEX_BACKEND", "faiss")

    visual_search = VisionAISearch(model_name=EMBEDDING_BACKEND, backend=INDEX_BACKEND)

    images = [
        ImageData(
            id=i,
            file_path=file_path
        ) for i, file_path in enumerate(sorted(glob("/home/appuser/src/archive/*.jpg")))
    ]

    visual_search.build(images)
    visual_search.save()

    query = "a cat"
    results = visual_search.search(query)
