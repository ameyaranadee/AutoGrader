from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('quiz', views.quizPage, name="quiz"),
    path('quiz/question/<str:pk>/', views.questionsPage, name="questions"),
    path('quiz/question/<str:pk>/success', views.successPage, name="success")    
]