import nltk
#nltk.download('punkt')
import csv
import sys
import re
import pandas as pd
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

haddad = ["haddad", "hadad", "andrade"]
marina = ["marina"]
bolsonaro = ["bolsonaro", "coiso", "bolso", "mito", "bonoro", "bostanaro", "salnorabo", "bobossauro", "bozonaro", "bolzonaro", "elenão", "elesim"]
boulos = ["boulos"]
ciro = ["ciro"]
daciolo = ["cabo", "daciolo"]
alckmin = ["alckmin"]
alvaro = ["alvaro"]
meirelles = ["meirelles", "meireles"]

dataset = pd.read_csv('DebateGlob.csv')
fullList = dataset.values.tolist()

for linha in fullList:
    teste = linha[1].lower()
    b = 0
    #for word in teste:
    if "haddad" in teste or "lula" in teste or "andrade" in teste or "manueladavila" in teste or "hadd" in teste:
        writerHaddad.writerow((linha[0], linha[1], linha[2]))
        b=1
    #break

    #for word in teste:
    if "marina" in teste:
        writerMarina.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break
    
    #for word in teste:
    if "bolsonaro" in teste or "coiso" in teste or "bolso" in teste or "mito" in teste or "bonoro" in teste or "bostanaro" in teste or "salnorabo" in teste or "bobossauro" in teste or "bozonaro" in teste or "elenão" in teste or "elenao" in teste or "elesim" in teste or "mourão" in teste or "mourao" in teste or "ditadura" in teste or "fraquejada" in teste or "facis" in teste or "nazis" in teste or "racis" in teste or "machis" in teste or "faca" in teste or "bozo" in teste or "boso" in teste:
        writerBolsonaro.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break

    #for word in teste:
    if "boulos" in teste:
        writerBolsonaro.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break

    #for word in teste:
    if "ciro" in teste or "cirao" in teste or "cirogomes" in teste:
        writerCiro.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break

    #for word in teste:
    if "daciolo" in teste  or "cabo" in teste:
        writerDaciolo.writerow((linha[0], linha[1], linha[2]))
        b=1
    #    break

    #for word in teste:
    if "alckmin" in teste:
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