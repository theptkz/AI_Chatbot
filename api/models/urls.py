from django.urls import path
import models.views as views

urlpatterns = [
    path('predict/', views.Intent_Model.as_view(), name = 'api_intent'),
]