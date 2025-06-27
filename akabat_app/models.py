from django.db import models

# Create your models here.

class SavedProject(models.Model):
    name = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    last_modified = models.DateTimeField(auto_now=True)
    user_uuid = models.CharField(max_length=100)
    folder_path = models.TextField()
    last_step = models.CharField(max_length=100, default="welcome")
    
    # NUEVOS CAMPOS
    session_csvs = models.JSONField(default=list)
    session_excluded = models.JSONField(default=dict)
    session_included = models.JSONField(default=dict)
