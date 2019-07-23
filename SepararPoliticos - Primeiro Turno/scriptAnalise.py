import numpy as np
import pandas as pd
import spacy
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

np.random.seed(500)
nlp = spacy.load('pt')

Corpus = pd.read_csv(r"DadosAnotados.csv",encoding='latin-1')

#PRÉ-PROCESSAMENTO
# Remover linhas vazias.
Corpus['Tweet'].dropna(inplace=True)

# Padroniza todo o texto em caixa baixa (letras minúsculas)
Corpus['Tweet'] = [entry.lower() for entry in Corpus['Tweet']]

#Remove nome de usuários (está diminuindo a precisão)
#Corpus['Tweet'] = [re.sub('@[^\s]+','', entry) for entry in Corpus['Tweet']]

# Remove @RT de retweets
Corpus['Tweet'] = [re.sub(r'^RT[\s]+', '', entry) for entry in Corpus['Tweet']]
    
# Remove hiperlinks
Corpus['Tweet'] = [re.sub(r'https?:\/\/.*[\r\n]*', '', entry) for entry in Corpus['Tweet']]

# Tokenização: Cada tweet é dividido em um array de palavras
Corpus['Tweet']= [word_tokenize(entry) for entry in Corpus['Tweet']]

# Remover stop-words e aplicar stemming
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Corpus['Tweet']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('portuguese') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Final_words = [token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB']
    Corpus.loc[index,'text_final'] = str(Final_words)
    #print([token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB'])
#print(token.lemma_ for token in Corpus['Tweet'] if token.pos_ == 'VERB')
#print(Corpus['text_final'])

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['Polaridade'],test_size=0.3)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

#Padrões TF-IDF
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)




#Alckmin

alckmin = pd.read_csv(r"listAlckmin.csv",encoding='latin-1')

#PRÉ-PROCESSAMENTO
# Remover linhas vazias.
alckmin['Mensagem'].dropna(inplace=True)

# Padroniza todo o texto em caixa baixa (letras minúsculas)
alckmin['Mensagem'] = [entry.lower() for entry in alckmin['Mensagem']]

#Remove nome de usuários (está diminuindo a precisão)
#alckmin['Mensagem'] = [re.sub('@[^\s]+','', entry) for entry in alckmin['Mensagem']]

# Remove @RT de reMensagems
alckmin['Mensagem'] = [re.sub(r'^RT[\s]+', '', entry) for entry in alckmin['Mensagem']]
    
# Remove hiperlinks
alckmin['Mensagem'] = [re.sub(r'https?:\/\/.*[\r\n]*', '', entry) for entry in alckmin['Mensagem']]

# Tokenização: Cada Mensagem é dividido em um array de palavras
alckmin['Mensagem']= [word_tokenize(entry) for entry in alckmin['Mensagem']]

# Remover stop-words e aplicar stemming
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(alckmin['Mensagem']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('portuguese') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Final_words = [token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB']
    alckmin.loc[index,'text_final'] = str(Final_words)
    #print([token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB'])
#print(token.lemma_ for token in alckmin['Mensagem'] if token.pos_ == 'VERB')
#print(alckmin['text_final'])

Test_X_TfidfAlckmin = Tfidf_vect.transform(alckmin['text_final'])

# ajustar o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NBTest = Naive.predict(Test_X_TfidfAlckmin)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
#print(Test_YRecover)

# Classificador SVM
# ajustar o conjunto de dados de treinamento no classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions = SVM.predict(Test_X_TfidfAlckmin)
predictions_recovered = Encoder.inverse_transform(predictions)

neutralAlckmin = 0
positiveAlckmin = 0
negativeAlckmin = 0

for predict in predictions_recovered:
    if (predict == "Neutro"):
        neutralAlckmin = neutralAlckmin + 1
    if (predict == "Positivo"):
        positiveAlckmin = positiveAlckmin + 1
    if (predict == "Negativo"):
        negativeAlckmin = negativeAlckmin + 1

print("Alckmin:")
print("Positivos: ", positiveAlckmin)
print("Neutros: ", neutralAlckmin)
print("Negativos: ", negativeAlckmin)
print("")

#Alvaro

Alvaro = pd.read_csv(r"listAlvaro.csv",encoding='latin-1')

#PRÉ-PROCESSAMENTO
# Remover linhas vazias.
Alvaro['Mensagem'].dropna(inplace=True)

# Padroniza todo o texto em caixa baixa (letras minúsculas)
Alvaro['Mensagem'] = [entry.lower() for entry in Alvaro['Mensagem']]

#Remove nome de usuários (está diminuindo a precisão)
#Alvaro['Mensagem'] = [re.sub('@[^\s]+','', entry) for entry in Alvaro['Mensagem']]

# Remove @RT de reMensagems
Alvaro['Mensagem'] = [re.sub(r'^RT[\s]+', '', entry) for entry in Alvaro['Mensagem']]
    
# Remove hiperlinks
Alvaro['Mensagem'] = [re.sub(r'https?:\/\/.*[\r\n]*', '', entry) for entry in Alvaro['Mensagem']]

# Tokenização: Cada Mensagem é dividido em um array de palavras
Alvaro['Mensagem']= [word_tokenize(entry) for entry in Alvaro['Mensagem']]

# Remover stop-words e aplicar stemming
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Alvaro['Mensagem']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('portuguese') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Final_words = [token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB']
    Alvaro.loc[index,'text_final'] = str(Final_words)
    #print([token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB'])
#print(token.lemma_ for token in Alvaro['Mensagem'] if token.pos_ == 'VERB')
#print(Alvaro['text_final'])

Test_X_TfidfAlvaro = Tfidf_vect.transform(Alvaro['text_final'])

# ajustar o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NBTest = Naive.predict(Test_X_TfidfAlvaro)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
#print(Test_YRecover)

# Classificador SVM
# ajustar o conjunto de dados de treinamento no classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions = SVM.predict(Test_X_TfidfAlvaro)
predictions_recovered = Encoder.inverse_transform(predictions)

neutralAlvaro = 0
positiveAlvaro = 0
negativeAlvaro = 0

for predict in predictions_recovered:
    if (predict == "Neutro"):
        neutralAlvaro = neutralAlvaro + 1
    if (predict == "Positivo"):
        positiveAlvaro = positiveAlvaro + 1
    if (predict == "Negativo"):
        negativeAlvaro = negativeAlvaro + 1

print("Alvaro:")
print("Positivos: ", positiveAlvaro)
print("Neutros: ", neutralAlvaro)
print("Negativos: ", negativeAlvaro)
print("")

#Amoedo

Amoedo = pd.read_csv(r"listAmoedo.csv",encoding='latin-1')

#PRÉ-PROCESSAMENTO
# Remover linhas vazias.
Amoedo['Mensagem'].dropna(inplace=True)

# Padroniza todo o texto em caixa baixa (letras minúsculas)
Amoedo['Mensagem'] = [entry.lower() for entry in Amoedo['Mensagem']]

#Remove nome de usuários (está diminuindo a precisão)
#Amoedo['Mensagem'] = [re.sub('@[^\s]+','', entry) for entry in Amoedo['Mensagem']]

# Remove @RT de reMensagems
Amoedo['Mensagem'] = [re.sub(r'^RT[\s]+', '', entry) for entry in Amoedo['Mensagem']]
    
# Remove hiperlinks
Amoedo['Mensagem'] = [re.sub(r'https?:\/\/.*[\r\n]*', '', entry) for entry in Amoedo['Mensagem']]

# Tokenização: Cada Mensagem é dividido em um array de palavras
Amoedo['Mensagem']= [word_tokenize(entry) for entry in Amoedo['Mensagem']]

# Remover stop-words e aplicar stemming
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Amoedo['Mensagem']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('portuguese') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Final_words = [token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB']
    Amoedo.loc[index,'text_final'] = str(Final_words)
    #print([token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB'])
#print(token.lemma_ for token in Amoedo['Mensagem'] if token.pos_ == 'VERB')
#print(Amoedo['text_final'])

Test_X_TfidfAmoedo = Tfidf_vect.transform(Amoedo['text_final'])

# ajustar o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NBTest = Naive.predict(Test_X_TfidfAmoedo)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
#print(Test_YRecover)

# Classificador SVM
# ajustar o conjunto de dados de treinamento no classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions = SVM.predict(Test_X_TfidfAmoedo)
predictions_recovered = Encoder.inverse_transform(predictions)

neutralAmoedo = 0
positiveAmoedo = 0
negativeAmoedo = 0

for predict in predictions_recovered:
    if (predict == "Neutro"):
        neutralAmoedo = neutralAmoedo + 1
    if (predict == "Positivo"):
        positiveAmoedo = positiveAmoedo + 1
    if (predict == "Negativo"):
        negativeAmoedo = negativeAmoedo + 1

print("Amoedo:")
print("Positivos: ", positiveAmoedo)
print("Neutros: ", neutralAmoedo)
print("Negativos: ", negativeAmoedo)
print("")

#Bolsonaro

Bolsonaro = pd.read_csv(r"listBolsonaro.csv",encoding='latin-1')

#PRÉ-PROCESSAMENTO
# Remover linhas vazias.
Bolsonaro['Mensagem'].dropna(inplace=True)

# Padroniza todo o texto em caixa baixa (letras minúsculas)
Bolsonaro['Mensagem'] = [entry.lower() for entry in Bolsonaro['Mensagem']]

#Remove nome de usuários (está diminuindo a precisão)
#Bolsonaro['Mensagem'] = [re.sub('@[^\s]+','', entry) for entry in Bolsonaro['Mensagem']]

# Remove @RT de reMensagems
Bolsonaro['Mensagem'] = [re.sub(r'^RT[\s]+', '', entry) for entry in Bolsonaro['Mensagem']]
    
# Remove hiperlinks
Bolsonaro['Mensagem'] = [re.sub(r'https?:\/\/.*[\r\n]*', '', entry) for entry in Bolsonaro['Mensagem']]

# Tokenização: Cada Mensagem é dividido em um array de palavras
Bolsonaro['Mensagem']= [word_tokenize(entry) for entry in Bolsonaro['Mensagem']]

# Remover stop-words e aplicar stemming
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Bolsonaro['Mensagem']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('portuguese') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Final_words = [token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB']
    Bolsonaro.loc[index,'text_final'] = str(Final_words)
    #print([token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB'])
#print(token.lemma_ for token in Bolsonaro['Mensagem'] if token.pos_ == 'VERB')
#print(Bolsonaro['text_final'])

Test_X_TfidfBolsonaro = Tfidf_vect.transform(Bolsonaro['text_final'])

# ajustar o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NBTest = Naive.predict(Test_X_TfidfBolsonaro)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
#print(Test_YRecover)

# Classificador SVM
# ajustar o conjunto de dados de treinamento no classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions = SVM.predict(Test_X_TfidfBolsonaro)
predictions_recovered = Encoder.inverse_transform(predictions)

neutralBolsonaro = 0
positiveBolsonaro = 0
negativeBolsonaro = 0

for predict in predictions_recovered:
    if (predict == "Neutro"):
        neutralBolsonaro = neutralBolsonaro + 1
    if (predict == "Positivo"):
        positiveBolsonaro = positiveBolsonaro + 1
    if (predict == "Negativo"):
        negativeBolsonaro = negativeBolsonaro + 1

print("Bolsonaro:")
print("Positivos: ", positiveBolsonaro)
print("Neutros: ", neutralBolsonaro)
print("Negativos: ", negativeBolsonaro)
print("")

#Boulos

Boulos = pd.read_csv(r"listBoulos.csv",encoding='latin-1')

#PRÉ-PROCESSAMENTO
# Remover linhas vazias.
Boulos['Mensagem'].dropna(inplace=True)

# Padroniza todo o texto em caixa baixa (letras minúsculas)
Boulos['Mensagem'] = [entry.lower() for entry in Boulos['Mensagem']]

#Remove nome de usuários (está diminuindo a precisão)
#Boulos['Mensagem'] = [re.sub('@[^\s]+','', entry) for entry in Boulos['Mensagem']]

# Remove @RT de reMensagems
Boulos['Mensagem'] = [re.sub(r'^RT[\s]+', '', entry) for entry in Boulos['Mensagem']]
    
# Remove hiperlinks
Boulos['Mensagem'] = [re.sub(r'https?:\/\/.*[\r\n]*', '', entry) for entry in Boulos['Mensagem']]

# Tokenização: Cada Mensagem é dividido em um array de palavras
Boulos['Mensagem']= [word_tokenize(entry) for entry in Boulos['Mensagem']]

# Remover stop-words e aplicar stemming
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Boulos['Mensagem']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('portuguese') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Final_words = [token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB']
    Boulos.loc[index,'text_final'] = str(Final_words)
    #print([token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB'])
#print(token.lemma_ for token in Boulos['Mensagem'] if token.pos_ == 'VERB')
#print(Boulos['text_final'])

Test_X_TfidfBoulos = Tfidf_vect.transform(Boulos['text_final'])

# ajustar o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NBTest = Naive.predict(Test_X_TfidfBoulos)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
#print(Test_YRecover)

# Classificador SVM
# ajustar o conjunto de dados de treinamento no classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions = SVM.predict(Test_X_TfidfBoulos)
predictions_recovered = Encoder.inverse_transform(predictions)

neutralBoulos = 0
positiveBoulos = 0
negativeBoulos = 0

for predict in predictions_recovered:
    if (predict == "Neutro"):
        neutralBoulos = neutralBoulos + 1
    if (predict == "Positivo"):
        positiveBoulos = positiveBoulos + 1
    if (predict == "Negativo"):
        negativeBoulos = negativeBoulos + 1

print("Boulos:")
print("Positivos: ", positiveBoulos)
print("Neutros: ", neutralBoulos)
print("Negativos: ", negativeBoulos)
print("")

#Ciro

Ciro = pd.read_csv(r"listCiro.csv",encoding='latin-1')

#PRÉ-PROCESSAMENTO
# Remover linhas vazias.
Ciro['Mensagem'].dropna(inplace=True)

# Padroniza todo o texto em caixa baixa (letras minúsculas)
Ciro['Mensagem'] = [entry.lower() for entry in Ciro['Mensagem']]

#Remove nome de usuários (está diminuindo a precisão)
#Ciro['Mensagem'] = [re.sub('@[^\s]+','', entry) for entry in Ciro['Mensagem']]

# Remove @RT de reMensagems
Ciro['Mensagem'] = [re.sub(r'^RT[\s]+', '', entry) for entry in Ciro['Mensagem']]
    
# Remove hiperlinks
Ciro['Mensagem'] = [re.sub(r'https?:\/\/.*[\r\n]*', '', entry) for entry in Ciro['Mensagem']]

# Tokenização: Cada Mensagem é dividido em um array de palavras
Ciro['Mensagem']= [word_tokenize(entry) for entry in Ciro['Mensagem']]

# Remover stop-words e aplicar stemming
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Ciro['Mensagem']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('portuguese') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Final_words = [token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB']
    Ciro.loc[index,'text_final'] = str(Final_words)
    #print([token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB'])
#print(token.lemma_ for token in Ciro['Mensagem'] if token.pos_ == 'VERB')
#print(Ciro['text_final'])

Test_X_TfidfCiro = Tfidf_vect.transform(Ciro['text_final'])

# ajustar o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NBTest = Naive.predict(Test_X_TfidfCiro)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
#print(Test_YRecover)

# Classificador SVM
# ajustar o conjunto de dados de treinamento no classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions = SVM.predict(Test_X_TfidfCiro)
predictions_recovered = Encoder.inverse_transform(predictions)

neutralCiro = 0
positiveCiro = 0
negativeCiro = 0

for predict in predictions_recovered:
    if (predict == "Neutro"):
        neutralCiro = neutralCiro + 1
    if (predict == "Positivo"):
        positiveCiro = positiveCiro + 1
    if (predict == "Negativo"):
        negativeCiro = negativeCiro + 1

print("Ciro:")
print("Positivos: ", positiveCiro)
print("Neutros: ", neutralCiro)
print("Negativos: ", negativeCiro)
print("")

#Daciolo

Daciolo = pd.read_csv(r"listDaciolo.csv",encoding='latin-1')

#PRÉ-PROCESSAMENTO
# Remover linhas vazias.
Daciolo['Mensagem'].dropna(inplace=True)

# Padroniza todo o texto em caixa baixa (letras minúsculas)
Daciolo['Mensagem'] = [entry.lower() for entry in Daciolo['Mensagem']]

#Remove nome de usuários (está diminuindo a precisão)
#Daciolo['Mensagem'] = [re.sub('@[^\s]+','', entry) for entry in Daciolo['Mensagem']]

# Remove @RT de reMensagems
Daciolo['Mensagem'] = [re.sub(r'^RT[\s]+', '', entry) for entry in Daciolo['Mensagem']]
    
# Remove hiperlinks
Daciolo['Mensagem'] = [re.sub(r'https?:\/\/.*[\r\n]*', '', entry) for entry in Daciolo['Mensagem']]

# Tokenização: Cada Mensagem é dividido em um array de palavras
Daciolo['Mensagem']= [word_tokenize(entry) for entry in Daciolo['Mensagem']]

# Remover stop-words e aplicar stemming
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Daciolo['Mensagem']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('portuguese') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Final_words = [token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB']
    Daciolo.loc[index,'text_final'] = str(Final_words)
    #print([token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB'])
#print(token.lemma_ for token in Daciolo['Mensagem'] if token.pos_ == 'VERB')
#print(Daciolo['text_final'])

Test_X_TfidfDaciolo = Tfidf_vect.transform(Daciolo['text_final'])

# ajustar o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NBTest = Naive.predict(Test_X_TfidfDaciolo)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
#print(Test_YRecover)

# Classificador SVM
# ajustar o conjunto de dados de treinamento no classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions = SVM.predict(Test_X_TfidfDaciolo)
predictions_recovered = Encoder.inverse_transform(predictions)

neutralDaciolo = 0
positiveDaciolo = 0
negativeDaciolo = 0

for predict in predictions_recovered:
    if (predict == "Neutro"):
        neutralDaciolo = neutralDaciolo + 1
    if (predict == "Positivo"):
        positiveDaciolo = positiveDaciolo + 1
    if (predict == "Negativo"):
        negativeDaciolo = negativeDaciolo + 1

print("Daciolo:")
print("Positivos: ", positiveDaciolo)
print("Neutros: ", neutralDaciolo)
print("Negativos: ", negativeDaciolo)
print("")

#Haddad

Haddad = pd.read_csv(r"listHaddad.csv",encoding='latin-1')

#PRÉ-PROCESSAMENTO
# Remover linhas vazias.
Haddad['Mensagem'].dropna(inplace=True)

# Padroniza todo o texto em caixa baixa (letras minúsculas)
Haddad['Mensagem'] = [entry.lower() for entry in Haddad['Mensagem']]

#Remove nome de usuários (está diminuindo a precisão)
#Haddad['Mensagem'] = [re.sub('@[^\s]+','', entry) for entry in Haddad['Mensagem']]

# Remove @RT de reMensagems
Haddad['Mensagem'] = [re.sub(r'^RT[\s]+', '', entry) for entry in Haddad['Mensagem']]
    
# Remove hiperlinks
Haddad['Mensagem'] = [re.sub(r'https?:\/\/.*[\r\n]*', '', entry) for entry in Haddad['Mensagem']]

# Tokenização: Cada Mensagem é dividido em um array de palavras
Haddad['Mensagem']= [word_tokenize(entry) for entry in Haddad['Mensagem']]

# Remover stop-words e aplicar stemming
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Haddad['Mensagem']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('portuguese') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Final_words = [token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB']
    Haddad.loc[index,'text_final'] = str(Final_words)
    #print([token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB'])
#print(token.lemma_ for token in Haddad['Mensagem'] if token.pos_ == 'VERB')
#print(Haddad['text_final'])

Test_X_TfidfHaddad = Tfidf_vect.transform(Haddad['text_final'])

# ajustar o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NBTest = Naive.predict(Test_X_TfidfHaddad)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
#print(Test_YRecover)

# Classificador SVM
# ajustar o conjunto de dados de treinamento no classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions = SVM.predict(Test_X_TfidfHaddad)
predictions_recovered = Encoder.inverse_transform(predictions)

neutralHaddad = 0
positiveHaddad = 0
negativeHaddad = 0

for predict in predictions_recovered:
    if (predict == "Neutro"):
        neutralHaddad = neutralHaddad + 1
    if (predict == "Positivo"):
        positiveHaddad = positiveHaddad + 1
    if (predict == "Negativo"):
        negativeHaddad = negativeHaddad + 1

print("Haddad:")
print("Positivos: ", positiveHaddad)
print("Neutros: ", neutralHaddad)
print("Negativos: ", negativeHaddad)
print("")

#Marina

Marina = pd.read_csv(r"listMarina.csv",encoding='latin-1')

#PRÉ-PROCESSAMENTO
# Remover linhas vazias.
Marina['Mensagem'].dropna(inplace=True)

# Padroniza todo o texto em caixa baixa (letras minúsculas)
Marina['Mensagem'] = [entry.lower() for entry in Marina['Mensagem']]

#Remove nome de usuários (está diminuindo a precisão)
#Marina['Mensagem'] = [re.sub('@[^\s]+','', entry) for entry in Marina['Mensagem']]

# Remove @RT de reMensagems
Marina['Mensagem'] = [re.sub(r'^RT[\s]+', '', entry) for entry in Marina['Mensagem']]
    
# Remove hiperlinks
Marina['Mensagem'] = [re.sub(r'https?:\/\/.*[\r\n]*', '', entry) for entry in Marina['Mensagem']]

# Tokenização: Cada Mensagem é dividido em um array de palavras
Marina['Mensagem']= [word_tokenize(entry) for entry in Marina['Mensagem']]

# Remover stop-words e aplicar stemming
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Marina['Mensagem']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('portuguese') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Final_words = [token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB']
    Marina.loc[index,'text_final'] = str(Final_words)
    #print([token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB'])
#print(token.lemma_ for token in Marina['Mensagem'] if token.pos_ == 'VERB')
#print(Marina['text_final'])

Test_X_TfidfMarina = Tfidf_vect.transform(Marina['text_final'])

# ajustar o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NBTest = Naive.predict(Test_X_TfidfMarina)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
#print(Test_YRecover)

# Classificador SVM
# ajustar o conjunto de dados de treinamento no classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions = SVM.predict(Test_X_TfidfMarina)
predictions_recovered = Encoder.inverse_transform(predictions)

neutralMarina = 0
positiveMarina = 0
negativeMarina = 0

for predict in predictions_recovered:
    if (predict == "Neutro"):
        neutralMarina = neutralMarina + 1
    if (predict == "Positivo"):
        positiveMarina = positiveMarina + 1
    if (predict == "Negativo"):
        negativeMarina = negativeMarina + 1

print("Marina:")
print("Positivos: ", positiveMarina)
print("Neutros: ", neutralMarina)
print("Negativos: ", negativeMarina)
print("")

#Meirelles

Meirelles = pd.read_csv(r"listMeirelles.csv",encoding='latin-1')

#PRÉ-PROCESSAMENTO
# Remover linhas vazias.
Meirelles['Mensagem'].dropna(inplace=True)

# Padroniza todo o texto em caixa baixa (letras minúsculas)
Meirelles['Mensagem'] = [entry.lower() for entry in Meirelles['Mensagem']]

#Remove nome de usuários (está diminuindo a precisão)
#Meirelles['Mensagem'] = [re.sub('@[^\s]+','', entry) for entry in Meirelles['Mensagem']]

# Remove @RT de reMensagems
Meirelles['Mensagem'] = [re.sub(r'^RT[\s]+', '', entry) for entry in Meirelles['Mensagem']]
    
# Remove hiperlinks
Meirelles['Mensagem'] = [re.sub(r'https?:\/\/.*[\r\n]*', '', entry) for entry in Meirelles['Mensagem']]

# Tokenização: Cada Mensagem é dividido em um array de palavras
Meirelles['Mensagem']= [word_tokenize(entry) for entry in Meirelles['Mensagem']]

# Remover stop-words e aplicar stemming
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Meirelles['Mensagem']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('portuguese') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Final_words = [token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB']
    Meirelles.loc[index,'text_final'] = str(Final_words)
    #print([token.lemma_ for token in nlp(str(Final_words)) if token.pos_ == 'VERB'])
#print(token.lemma_ for token in Meirelles['Mensagem'] if token.pos_ == 'VERB')
#print(Meirelles['text_final'])

Test_X_TfidfMeirelles = Tfidf_vect.transform(Meirelles['text_final'])

# ajustar o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NBTest = Naive.predict(Test_X_TfidfMeirelles)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
#print(Test_YRecover)

# Classificador SVM
# ajustar o conjunto de dados de treinamento no classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions = SVM.predict(Test_X_TfidfMeirelles)
predictions_recovered = Encoder.inverse_transform(predictions)

neutralMeirelles = 0
positiveMeirelles = 0
negativeMeirelles = 0

for predict in predictions_recovered:
    if (predict == "Neutro"):
        neutralMeirelles = neutralMeirelles + 1
    if (predict == "Positivo"):
        positiveMeirelles = positiveMeirelles + 1
    if (predict == "Negativo"):
        negativeMeirelles = negativeMeirelles + 1

print("Meirelles:")
print("Positivos: ", positiveMeirelles)
print("Neutros: ", neutralMeirelles)
print("Negativos: ", negativeMeirelles)