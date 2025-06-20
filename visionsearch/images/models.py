from django.db import models

# Create your models here.
class ImageAsset(models.Model):
    title = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)
    file_path = models.CharField(max_length=512, unique=True)  # Path or URL to the image
    embedding_index = models.IntegerField(null=True, blank=True)  # Position in FAISS index
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title or f"Image #{self.id}"