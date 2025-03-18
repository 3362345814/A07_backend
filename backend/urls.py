from django.urls import path
from . import views

urlpatterns = [
    path('process-images/', views.process_medical_images, name='process_images'),
]