from django.urls import path
import models.views as views

urlpatterns = [
    path('predict/', views.Models.as_view(), name = 'api'),
]