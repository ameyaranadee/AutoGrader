from django.db import models

# Create your models here.
class Question(models.Model):
    """ A model of the 8 questions. """
    id = models.IntegerField(primary_key=True)
    question_title = models.TextField(max_length=100000)
    set = models.IntegerField(unique=True)
    min_score = models.IntegerField()
    max_score = models.IntegerField()

    def __str__(self):
        return str(self.set)

class Answer(models.Model):
    """ Answer to be submitted. """
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    content = models.TextField(max_length=100000)
    score = models.IntegerField(null=True, blank=True)