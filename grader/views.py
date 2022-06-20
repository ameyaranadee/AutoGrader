from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
import random

from .forms import AnswerForm
from .models import Question, Answer

# Create your views here.

# questions = [
#     {'id': 1, 'title': 'How to create an app in Django?'}, 
#     {'id': 2, 'title': 'How to set up a project in Node.js?'}, 
#     {'id': 3, 'title': 'Is React.js compatible with Django?'}
# ]

def home(request):
    questions = Question.objects.all()

    context = {'questions': questions}
    return render(request, 'grader/home.html', context)

def successPage(request):
    return render(request, 'grader/success.html')

def questionsPage(request, pk):
    question = Question.objects.get(id=pk)
    form = AnswerForm
    # for q in questions:
    #     if q['id'] == int(pk):
    #         question = q

    if request.method == 'POST':
        form = AnswerForm(request.POST)
        if form.is_valid():
            content = form.cleaned_data.get('answer')
            score = random.randint(0,10)

            # auto grading nlp code

            answer = Answer.objects.create(
                content=content,
                score = score,
                question=question,
            )
        context = {'answer': answer}
        return render(request, 'grader/success.html', context)
    else:
        form = AnswerForm()

    context = {'question': question, 'form': form}
    return render(request, 'grader/question.html', context)