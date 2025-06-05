from django.urls import path

from disease_prediction import views

urlpatterns = [
    path("", views.index, name="index"),
]
