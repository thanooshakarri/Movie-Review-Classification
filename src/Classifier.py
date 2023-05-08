import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from scipy.sparse import csr_matrix, hstack
import langid


class Classifier:
  #classifier to classify text 
  def __init__(self):
    #initializing a Featurization class, LabelEncoder and Logistic Regression
    self.feature_obj=Featurization()
    self.Logistic_R=LogisticRegression(class_weight="balanced",C=10,solver="liblinear",penalty="l2")
    self.le=LabelEncoder()
  def fit(self,data,label):
    #this method is used to fit the data
    data["TEXT"]=data["TEXT"].astype(str)
    pre_processed_data=self.feature_obj.get_features(data)
    self.Logistic_R.fit(pre_processed_data,self.le.transform(label))
  def predict(self,data):
    #this method is used to predict classifier with model
    pre_processed_data=self.feature_obj.get_features(data)
    return self.Logistic_R.predict(pre_processed_data)
  def f1_score(self,y_test,y_pred):
    #calculating macro f1 score
    return metrics.f1_score(y_test,y_pred,average="macro")

class Featurization:
  #Featurization class=pre_processing+data featurization
  def __init__(self):
    #initializing TfidfVectorizer
    self.cv=TfidfVectorizer(  
           ngram_range=(1,2), stop_words="english", lowercase=False)
  def language(self,text):
    #creating new feature to specify language
    l=langid.classify(text)[0]
    if(l!="en"):
      return 0
    return 1
  def remove_HTML(self,text):
    #removing html tags
    return re.sub("<[^<]+?>","",text)
  def lem(self,text):
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    # Tokenize the sentence
    words = nltk.word_tokenize(text)
    # Lemmatize the words
    return " ".join([lemmatizer.lemmatize(word) for word in words])
  def get_features(self,data):
    # Concatenate the two matrices horizontally-features
    data=self.pre_process(data)
    self.cv.fit(data)
    return hstack([self.cv.transform(data["TEXT"]), csr_matrix(data["lang"]).transpose()])
  def pre_process(self,data):
    #data pre_processing
    data["TEXT"]=data["TEXT"].astype(str)
    data["TEXT"] = data["TEXT"].apply(self.remove_HTML)
    data["lang"]=data["TEXT"].apply(self.language)
    data["TEXT"] = data["TEXT"].apply(self.lem)
    return data