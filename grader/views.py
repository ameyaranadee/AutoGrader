from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
import random

from .forms import AnswerForm
from .models import Question, Answer, Set

# required libraries
import numpy as np
from numpy.linalg import norm
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import word_tokenize, sent_tokenize
import scipy
from scipy import spatial
import re
nltk.download('punkt')
wpt = nltk.WordPunctTokenizer()
stop_words = stopwords.words('english')

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

nltk.download('wordnet')
nltk.download('omw-1.4')

import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')


dataset = pd.read_csv('E:/Internships/C-MInDS/Auto Grading Django/AutoGrader/data/Data.csv')
dataset_groups = dataset.groupby('Questions')

def cosine(u, v):
    return (np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

def computeVecSum(vectors):
    n = len(vectors)  # vectors contains embedding score for each lemma word in sent
    d = 399 #length of embeddings of lemma word for baroni et al embeddings

    s = []
    for i in range(d):
        s.append(0)
   

    s = np.array(s)
    
    for vec in vectors:
        s = s + np.array(vec)

    return (s)

def load_data(FileName= 'E:/Internships/C-MInDS/Auto Grading Django/AutoGrader/data/EN-wform.w.5.cbow.neg10.400.subsmpl.txt'):
    
    embeddings = {}
    file = open(FileName,'r', encoding="utf8")
    i = 0
    print("Loading word embeddings first time")
    for line in file:
        # print line

        tokens = line.split('\t')

        #since each line's last token content '\n'
        # we need to remove that
        tokens[-1] = tokens[-1].strip()

        #each line has 400 tokens
        for i in range(1, len(tokens)):
            tokens[i] = float(tokens[i])
            
        embeddings[tokens[0]] = tokens[1:-1]
    print("finished")
    return embeddings
e = load_data()

punctuations = ['(',')','-lrb-','.',',','-','?','!',';','_',':','{','}','[','/',']','...','"','\'', '>', '<','-rrb-']  

def normalize_document(doc):
  doc = re.sub(r'[^a-zA-Z\s][<br><br>][<br>]', '', doc, re.I|re.A)
  doc = doc.lower()
  doc = doc.strip()
  tokens = wpt.tokenize(doc)
  filter_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words] # and token not in punctuations]
  doc = ' '.join(filter_tokens)
  return doc

def normalize_documentQD(doc, que):
  doc = re.sub(r'[^a-zA-Z\s][<br><br>][<br>]', '', doc, re.I|re.A)
  doc = doc.lower()
  doc = doc.strip()
  tokens = wpt.tokenize(doc)
  q = wpt.tokenize(que)
  filter_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token not in q]
  doc = ' '.join(filter_tokens)
  return doc

df = pd.read_csv('E:/Internships/C-MInDS/Auto Grading Django/AutoGrader/data/Data_bert_Cos_Wm.csv')
df = df.drop(['Unnamed: 0'], axis = 1) 
dfn = df.dropna()
X = dfn.drop(['Score'], axis = 1)
y = dfn.Score

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state = 12)

ridge=Ridge(alpha=0.1, normalize=True)
ridge_mod = ridge.fit(xtrain,ytrain)
ypred = ridge_mod.predict(xtest)
score = ridge_mod.score(xtest,ytest)
mse = mean_squared_error(ytest,ypred)
corr, _ = pearsonr(ytest, ypred)
print('Pearsons correlation: %.3f' % corr)
print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
   .format(score, mse,np.sqrt(mse)))

# Cos_Wm

def cos_Wm(score, responses_s, response_s, ans):
  sim = []
  sc = []
  for i in range(len(score)):
    if score[i] == 5:
      sim_5 = cosine(response_s, responses_s[i])
      sim.append(sim_5)
      sc.append(5)
  cos = ((sum(sc))*cosine(ans, response_s) + sum(sim))/(sum(sc)*2)
    
  return cos

# alignment
pair = []
x = []
y = []
def Alignment(sent_a, sent_b):
  tokens_a = word_tokenize(sent_a)
  tokens_b = word_tokenize(sent_b)
  for token in tokens_a:
    if token in e.keys():
            vec1 = e[token]
            for word in tokens_b:
              if word in e.keys():
                vec2 = e[word]
                sim_word = cosine(vec1, vec2)
                if sim_word >=0.40:
                  pair.append((word, token))
                  x.append(word)
                  y.append(token)

  a = (len(set(x))+len(set(y)))/((len(set(sent_a))+ len(set(sent_b))))
  return a

# Create your views here
def home(request):
    sets = Set.objects.all()
    questions = Question.objects.all()
    
    context = {'sets': sets, 'questions': questions}
    return render(request, 'grader/home.html', context)

def successPage(request):
    return render(request, 'grader/success.html')

def quizPage(request):
    questions = Question.objects.all()

    context = {'questions': questions}
    return render(request, 'grader/quiz.html', context)

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
            
            model_answer = question.model_answer
            # auto grading nlp code
            data = dataset_groups.get_group(question.question_title)
            
            # Preprocessing
            answer_p = normalize_document(model_answer)
            response_p = normalize_document(content)
            response_p_qd = normalize_documentQD(content, question.question_title)

            answer_emd = sbert_model.encode(answer_p)
            response_emd = sbert_model.encode(response_p)
            response_emd_qd = sbert_model.encode(response_p_qd)
            
            qd_responses = [normalize_documentQD(r, question.question_title) for r in data.Texts]

            responses_emd = sbert_model.encode(list(map(str, data.Texts)))
            qd_responses = [normalize_documentQD(r, question.question_title) for r in data.Texts]
            responses_emd_qd = sbert_model.encode(list(map(str, qd_responses)))

            # Cos_feature
            cos = cos_Wm(list(data.Score), responses_emd, response_emd, answer_emd)
            cos_qd = cos_Wm(list(data.Score), responses_emd_qd, response_emd_qd, answer_emd)

            alignment_normal = Alignment(answer_p, response_p)
            alignment_qd = Alignment(answer_p, response_p_qd)

            Dist = norm(answer_emd - response_emd)
            DistQD = norm(answer_emd - response_emd_qd)

            Lans = len(word_tokenize(answer_p))
            Lres = len(word_tokenize(response_p))
            Length_ratio = Lres/Lans

            ratio = (fuzz.ratio(answer_p, response_p))/100
            ratioqd = (fuzz.ratio(answer_p, response_p))/100

            pratio = (fuzz.partial_ratio(answer_p, response_p_qd))/100
            pratioqd = (fuzz.partial_ratio(answer_p, response_p_qd))/100

            tsoratio = (fuzz.token_sort_ratio(answer_p, response_p))/100
            tsoratioqd = (fuzz.token_sort_ratio(answer_p, response_p_qd))/100

            tseratio = (fuzz.token_set_ratio(answer_p, response_p))/100
            tseratioqd = (fuzz.token_set_ratio(answer_p, response_p))/100

            array = np.array([cos, alignment_normal, Length_ratio, Dist, cos_qd, alignment_qd, DistQD, ratio, ratioqd, pratio, pratioqd, tsoratio, tsoratioqd, tseratio, tseratioqd])
            array = array.reshape(1, -1)
            ypred = ridge_mod.predict(array)
            if ypred[0]>5.0: ypred[0] = 5.0
            if ypred[0]<0.0: ypred[0] = 0.0
            
            # score = random.randint(0,10)

            # auto grading nlp code

            answer = Answer.objects.create(
                content=content,
                score = ypred[0],
                question=question,
            )
        context = {'answer': answer}
        return render(request, 'grader/success.html', context)
    else:
        form = AnswerForm()

    context = {'question': question, 'form': form}
    return render(request, 'grader/question.html', context)