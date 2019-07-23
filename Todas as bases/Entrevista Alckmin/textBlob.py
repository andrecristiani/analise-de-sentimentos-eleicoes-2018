import nltk
#nltk.download('punkt')
import csv
import sys
import re
import pandas as pd
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from nltk.corpus import stopwords

listTraning = []
listAnalisys = []
listTests = []
classifiedList = []
datasetTraining = pd.read_csv('EntrevistaAlckminTrain.csv', usecols=["Mensagem", "Sentimento"])
datasetTest = pd.read_csv('EntrevistaAlckminTest.csv', usecols=["Mensagem", "Sentimento"])
datasetAnalisys = pd.read_csv('EntrevistaAlckminAnalisys.csv', usecols=["Mensagem"])
test_set = datasetTest.values.tolist()
training_set = datasetTraining.values.tolist()
analisys_set = datasetAnalisys.values.tolist()

positivo = 0
negativo = 0
neutro = 0
ambiguo = 0

for linha in training_set:
    if(isinstance(linha[0], str)):
        linha[0] = re.sub(r"http\S+", "", str(linha[0]))
        index = linha[0].find(':')
        index = index+1
        linha[0] = linha[0][index:]
        linha[0] = linha[0].lower()
        linha[0] = re.sub(r"//\S+", "", str(linha[0]))
        linha[0] = re.sub(r"@/\S+", "", str(linha[0]))
        listTraning.append((linha[0], linha[1]))

for linha in test_set:
    if(isinstance(linha[0], str)):
        linha[0] = re.sub(r"http\S+", "", str(linha[0]))
        index = linha[0].find(':')
        index = index+1
        linha[0] = linha[0][index:]
        linha[0] = linha[0].lower()
        linha[0] = re.sub(r"//\S+", "", str(linha[0]))
        linha[0] = re.sub(r"@/\S+", "", str(linha[0]))
        listTests.append((linha[0], linha[1]))

for linha in analisys_set:
    if(isinstance(linha[0], str)):
        linha[0] = re.sub(r"http\S+", "", str(linha[0]))
        index = linha[0].find(':')
        index = index+1
        linha[0] = linha[0][index:]
        linha[0] = linha[0].lower()
        linha[0] = re.sub(r"//\S+", "", str(linha[0]))
        linha[0] = re.sub(r"@/\S+", "", str(linha[0]))
        listAnalisys.append(linha[0])

stopwordsnltk = nltk.corpus.stopwords.words('portuguese')

def applyChanges(text):
    text = re.sub(r"http\S+", "", str(text))
    index = text.find(':')
    index = index+1
    text = text[index:]
    text = text.lower()
    text = re.sub(r"//\S+", "", str(text))
    text = re.sub(r"@/\S+", "", str(text))
    return text
        
cl = NaiveBayesClassifier(listTraning)
accuracy = cl.accuracy(listTests)

for linha in listAnalisys:
    blob = TextBlob(linha,classifier=cl)
    if blob.classify() == "Positivo":
            positivo = positivo + 1

    if blob.classify() == "Negativo":
            negativo = negativo + 1

    if blob.classify() == "Neutro":
            neutro = neutro + 1

    if blob.classify() == "Ambíguo":
            ambiguo = ambiguo + 1

for linha in listTraning:
    if linha[1] == "Positivo":
            positivo = positivo + 1

    if linha[1] == "Negativo":
            negativo = negativo + 1

    if linha[1] == "Neutro":
            neutro = neutro + 1

    if linha[1] == "Ambíguo":
            ambiguo = ambiguo + 1

for linha in listTests:
    if linha[1] == "Positivo":
            positivo = positivo + 1

    if linha[1] == "Negativo":
            negativo = negativo + 1

    if linha[1] == "Neutro":
            neutro = neutro + 1

    if linha[1] == "Ambíguo":
            ambiguo = ambiguo + 1

print('Precisão da previsão:{}'.format(accuracy))
print("Positivo", positivo)
print("Negativo", negativo)
print("Neutro", neutro)
print("Ambíguo", ambiguo)