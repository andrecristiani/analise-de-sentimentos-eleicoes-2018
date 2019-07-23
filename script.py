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

#print(Tfidf_vect.vocabulary_)

#print(Train_X_Tfidf)

# ajustar o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_NB = Naive.predict(Test_X_Tfidf)

#imprimir na tela a acurácia, f-measure, precisão e recall
print("Naive Bayes Accuracy: ",accuracy_score(predictions_NB, Test_Y)*100)
print("Naive Bayes F-Measure: ",f1_score(predictions_NB, Test_Y, average="macro")*100)
print("Naive Bayes Precision: ",precision_score(predictions_NB, Test_Y, average="macro")*100)
print("Naive Bayes Recall: ",recall_score(predictions_NB, Test_Y, average="macro")*100)

# Classificador SVM
# ajustar o conjunto de dados de treinamento no classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
#imprimir na tela a acurácia, f-measure, precisão e recall
print("SVM Accuracy: ",accuracy_score(predictions_SVM, Test_Y)*100)
print("SVM F-Measure: ",f1_score(predictions_SVM, Test_Y, average="macro")*100)
print("SVM Precision: ",precision_score(predictions_SVM, Test_Y, average="macro")*100)
print("SVM Recall: ",recall_score(predictions_SVM, Test_Y, average="macro")*100)
