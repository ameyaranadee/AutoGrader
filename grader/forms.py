from django import forms
from .models import Question, Answer

class AnswerForm(forms.ModelForm):
    answer = forms.CharField(max_length=100000, widget=forms.Textarea(attrs={'class': "form-control",
                'style': 'max-width: 800px;',
                'placeholder': 'Name', 'rows': 5, 'placeholder': "What's on your mind?"}))

    class Meta:
        model = Answer
        fields = ['answer']