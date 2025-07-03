from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from common_utils.aisearch.core import VisionAISearch
from common_utils.indexing.types import ImageData
from glob import glob
import os

class SearchApp:
    """
    A FastAPI-based web interface for semantic image search with natural language queries.
    """

    def __init__(self, data: str = "images", device: str = None):
        """
        Initialize the SearchApp with VisualAISearch backend.

        Args:
            data (str): Directory of indexed images.
            device (str): Device for embedding ("cpu", "cuda").
        """
        self.template = "index3.html"
        self.searcher = VisionAISearch(index_path="/media/faiss.index", model_name="clip")

        self.images = [
            ImageData(
                id=i,
                file_path=file_path
            ) for i, file_path in enumerate(sorted(glob(f"{data}/*.jpg")))
        ]

        self.searcher.build(images=self.images)
        self.searcher.save()
        self.app = FastAPI(title="VisionSearch Semantic Image Search")

        # Setup templates and static files
        templates_path = Path("templates")
        static_path = Path(data).resolve()

        self.templates = Jinja2Templates(directory=str(templates_path))
        self.app.mount("/images", StaticFiles(directory=static_path), name="images")

        # Define routes
        self.app.add_api_route("/", self.index, methods=["GET"], response_class=HTMLResponse)
        self.app.add_api_route("/", self.search, methods=["POST"], response_class=HTMLResponse)

    async def index(self, request: Request):
        """Render the empty search form."""
        return self.templates.TemplateResponse(self.template, {"request": request, "results": []})

    async def search(self, request: Request, query: str = Form(...)):
        """Handle search query and render results."""
        results = self.searcher.search(query, k=30)

        print(results)
        images = [
            {
                "path": os.path.basename(self.images[int(i)].file_path),
                "score": score,
            } for i, score in results
            ]
        return self.templates.TemplateResponse(self.template, {"request": request, "results": images, 'query': query})

    def get_app(self):
        """Return FastAPI app instance."""
        return self.app


search_app = SearchApp(data="/home/appuser/src/archive")
app = search_app.get_app()
