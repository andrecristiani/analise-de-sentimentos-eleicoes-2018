import nltk
#nltk.download('punkt')
import csv
import sys
import re
import pandas as pd
import numpy
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from nltk.corpus import stopwords

arquivoHaddad = open('listHaddad.csv', 'w')
writerHaddad = csv.writer(arquivoHaddad)
writerHaddad.writerow(['Usuario','Mensagem', 'Localizacao'])

arquivoBolsonaro = open('listBolsonaro.csv', 'w')
writerBolsonaro = csv.writer(arquivoBolsonaro)
writerBolsonaro.writerow(['Usuario','Mensagem', 'Localizacao'])

listHaddad = []
listMarina = []
listBolsonaro = []
listBoulos = []
listCiro = []
listDaciolo = []
listAlckmin = []
listAlvaro = []
listMeirelles = []
listAmoedo = []
listNon = []

haddad = ["haddad", "hadad", "andrade", "haddadpresidente", "haddad13", "agoraéhaddad", "haddaptando", "13neles", "ptsim", "haddad13", "viravirouhaddad"]
marina = ["marina"]
bolsonaro = ["bolsonaro", "coiso", "bolso", "mito", "bonoro", "bostanaro", "salnorabo", "bobossauro", "bozonaro", "bolzonaro", "elenão", "elesim", "bolsonaro17", "17neles", "b17", "jair", "bolsolixo", "nazistas", "obrasilvota17"]
boulos = ["boulos", "Boulos"]
ciro = ["ciro", "viraviraclr0", "ciro12", "viraviraciro12"]
daciolo = ["cabo", "daciolo", "deux", "deuxx"]
alckmin = ["alckmin", "xuxu", "chuchu"]
alvaro = ["alvaro"]
meirelles = ["meirelles", "meireles"]

dataset = pd.read_csv('2Turno.csv')
fullList = dataset.values.tolist()

for linha in fullList:
    teste = linha[1].lower()
    b = 0
    if "haddad" in teste or "lula" in teste or "andrade" in teste or "manueladavila" in teste or "hadd" in teste or "haddad" in teste or "hadad" in teste or "andrade" in teste or "haddadpresidente" in teste or "haddad13" in teste or "agoraéhaddad" in teste or "haddaptando" in teste or "13neles" in teste or "ptsim" in teste or "haddad13" in teste or "viravirouhaddad" in teste:
        writerHaddad.writerow((linha[0], linha[1], linha[2]))
        b=1
    #break
    
    #for word in teste:
    if "bolsonaro" in teste or "coiso" in teste or "bolso" in teste or "mito" in teste or "bonoro" in teste or "bostanaro" in teste or "salnorabo" in teste or "bobossauro" in teste or "bozonaro" in teste or "bolzonaro" in teste or "elenão" in teste or "elesim" in teste or "bolsonaro17" in teste or "17neles" in teste or "b17" in teste or "jair" in teste or "bolsolixo" in teste or "nazistas" in teste or "obrasilvota17" in teste:
        writerBolsonaro.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break
    
print("Candidatos separados com sucesso!")
