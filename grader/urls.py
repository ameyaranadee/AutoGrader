from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('question/<str:pk>/', views.questionsPage, name="questions"),
    path('question/<str:pk>/success', views.successPage, name="success"),
]