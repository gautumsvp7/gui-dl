"""
crowd_app/models.py
-------------------
No database models are needed for the upload / results workflow.

The upload view stores the uploaded file directly on disk under
MEDIA_ROOT/uploads/ and passes the public URL to the results view
via the Django session (server-side cookie-backed key–value store).
This avoids creating a migration for a model that would otherwise
just be a one-column table of file paths.

If you later want to keep an audit log of every uploaded image and
its predicted crowd count you can add a model here, e.g.:

    from django.db import models

    class CrowdPrediction(models.Model):
        uploaded_at   = models.DateTimeField(auto_now_add=True)
        image         = models.ImageField(upload_to='uploads/')
        density_map   = models.ImageField(upload_to='results/', blank=True)
        crowd_count   = models.IntegerField(null=True)
        mae           = models.FloatField(null=True)
        mse           = models.FloatField(null=True)
        model_name    = models.CharField(max_length=64, blank=True)

        def __str__(self):
            return f'Prediction #{self.pk} – count {self.crowd_count}'

Then run:
    python manage.py makemigrations crowd_app
    python manage.py migrate
"""
