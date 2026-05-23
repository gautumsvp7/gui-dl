"""
crowd_app/urls.py
-----------------
URL configuration for the CrowdVision upload / results workflow.

WHY app_name?
  Django's URL namespacing requires the app_name variable (or the
  'namespace' kwarg in include()) to be set so that {% url %} tags
  can use the 'crowd_app:upload' / 'crowd_app:results' form.
  This prevents name collisions if another app also defines a view
  called 'upload'.

These routes are mounted under /crowd/ in mysite/urls.py:
    path('crowd/', include('crowd_app.urls', namespace='crowd_app'))

So the full URLs are:
    /crowd/upload/   →  upload_view   (GET + POST)
    /crowd/results/  →  results_view  (GET)
"""

from django.urls import path
from . import views

app_name = 'crowd_app'

urlpatterns = [
    path('upload/',      views.upload_view,      name='upload'),
    path('results/',     views.results_view,     name='results'),
    path('live/',        views.live_view,         name='live'),
    path('live-infer/',  views.live_infer_view,   name='live_infer'),
]
