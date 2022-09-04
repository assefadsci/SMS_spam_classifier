# import libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

#read the data in
df= pd.read_csv('spam.csv', usecols= ['v1', 'v2'], encoding="latin-1")

#rename column names
df.rename(columns= {'v1': 'label', 'v2':'sms'}, inplace= True)

#Tokenizing, removing stop words and Lemmatization
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def CleanSms(sms):
    #Replacing all non-alphabetic characters with a space
    sms = re.sub('[^a-zA-Z]', ' ', sms)
    #converting to lowercase
    sms = sms.lower()
    #tokenization
    sms_tokens= nltk.word_tokenize(sms)
    #Removing Stopwords
    nostopword_sms = [token for token in sms_tokens if token not in stop_words]
    #Lemmatization
    lemmatized_sms = [lemmatizer.lemmatize(token, pos ='v') for token in nostopword_sms]

    return lemmatized_sms

# creating new features
df['Transformed_sms']= df['sms'].apply(CleanSms)

#Creating a corpus of text feature to encode further into vectorized form
corpus= []
for i in df["Transformed_sms"]:
    msg = ' '.join([row for row in i])
    corpus.append(msg)

#Changing text data in to numbers
vectorize = TfidfVectorizer()
X = vectorize.fit_transform(corpus).toarray()

# Setup features and label as X and y
y= df['label']

#train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
classifier = LogisticRegression(C= 10, random_state=42)
classifier.fit(X_train, y_train)

#Serializing the model
pickle.dump(classifier, open('classifier.pkl', 'wb'))

#Serializing the vectorizer
pickle.dump(vectorize, open('vectorize.pkl', 'wb'))
