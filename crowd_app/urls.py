from django.urls import path
from . import views

app_name = 'crowd_app'

urlpatterns = [
    path('upload/', views.upload_view, name='upload'),
    path('results/', views.results_view, name='results'),
    path('live/', views.live_view, name='live'),
    path('live-infer/', views.live_infer_view, name='live_infer'),
]
