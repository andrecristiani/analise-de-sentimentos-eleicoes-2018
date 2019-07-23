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

arquivoMarina = open('listMarina.csv', 'w')
writerMarina = csv.writer(arquivoMarina)
writerMarina.writerow(['Usuario','Mensagem', 'Localizacao'])

arquivoBolsonaro = open('listBolsonaro.csv', 'w')
writerBolsonaro = csv.writer(arquivoBolsonaro)
writerBolsonaro.writerow(['Usuario','Mensagem', 'Localizacao'])

arquivoBoulos = open('listBoulos.csv', 'w')
writerBoulos = csv.writer(arquivoBoulos)
writerBoulos.writerow(['Usuario','Mensagem', 'Localizacao'])

arquivoCiro = open('listCiro.csv', 'w')
writerCiro = csv.writer(arquivoCiro)
writerCiro.writerow(['Usuario','Mensagem', 'Localizacao'])

arquivoDaciolo = open('listDaciolo.csv', 'w')
writerDaciolo = csv.writer(arquivoDaciolo)
writerDaciolo.writerow(['Usuario','Mensagem', 'Localizacao'])

arquivoAlckmin = open('listAlckmin.csv', 'w')
writerAlckmin = csv.writer(arquivoAlckmin)
writerAlckmin.writerow(['Usuario','Mensagem', 'Localizacao'])

arquivoAlvaro = open('listAlvaro.csv', 'w')
writerAlvaro = csv.writer(arquivoAlvaro)
writerAlvaro.writerow(['Usuario','Mensagem', 'Localizacao'])

arquivoMeirelles = open('listMeirelles.csv', 'w')
writerMeirelles = csv.writer(arquivoMeirelles)
writerMeirelles.writerow(['Usuario','Mensagem', 'Localizacao'])

arquivoAmoedo = open('listAmoedo.csv', 'w')
writerAmoedo = csv.writer(arquivoAmoedo)
writerAmoedo.writerow(['Usuario','Mensagem', 'Localizacao'])

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

dataset = pd.read_csv('DiaEleiçãoCompleto.csv')
fullList = dataset.values.tolist()

for linha in fullList:
    teste = linha[1].lower()
    b = 0
    if "haddad" in teste or "lula" in teste or "andrade" in teste or "manueladavila" in teste or "hadd" in teste or "haddad" in teste or "hadad" in teste or "andrade" in teste or "haddadpresidente" in teste or "haddad13" in teste or "agoraéhaddad" in teste or "haddaptando" in teste or "13neles" in teste or "ptsim" in teste or "haddad13" in teste or "viravirouhaddad" in teste:
        writerHaddad.writerow((linha[0], linha[1], linha[2]))
        b=1
    #break

    #for word in teste:
    if "marina" in teste:
        writerMarina.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break
    
    #for word in teste:
    if "bolsonaro" in teste or "coiso" in teste or "bolso" in teste or "mito" in teste or "bonoro" in teste or "bostanaro" in teste or "salnorabo" in teste or "bobossauro" in teste or "bozonaro" in teste or "bolzonaro" in teste or "elenão" in teste or "elesim" in teste or "bolsonaro17" in teste or "17neles" in teste or "b17" in teste or "jair" in teste or "bolsolixo" in teste or "nazistas" in teste or "obrasilvota17" in teste:
        writerBolsonaro.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break

    #for word in teste:
    if "boulos" in teste:
        writerBoulos.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break

    #for word in teste:
    if "ciro" in teste or "cirao" in teste or "cirogomes" in teste or "ciro" in teste or "viraviraclr0" in teste or "ciro12" in teste or "viraviraciro12" in teste:
        writerCiro.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break

    #for word in teste:
    if "daciolo" in teste  or "cabo" in teste or "deux" in teste or "deuxx" in teste:
        writerDaciolo.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break

    #for word in teste:
    if "alckmin" in teste or "xuxu" in teste or "chuchu" in teste:
        writerAlckmin.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break

    #for word in teste:
    if "alvaro" in teste or "álvaro" in teste:
        writerAlvaro.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break

    #for word in teste:
    if "meirelles" in teste or "meireles" in teste:
        writerMeirelles.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break

    #for word in teste:
    if "amoedo" in teste or "amoêdo" in teste:
        writerAmoedo.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break
    
print("Candidatos separados com sucesso!")
