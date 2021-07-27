
# Importing libraries
import re, nltk
import time
import numpy as np
import pandas as pd
import csv
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

#Normalising dtext data to make it usefull
def normalizer(Review):
    soup = BeautifulSoup(Review, 'lxml')# removing HTML encoding#
    souped = soup.get_text()
    only_words = re.sub("([0-9]+)|('&#039;')|(\w+:\/\\\S+)"," ", souped)## Removing numbers and &#039;
    tokens = nltk.word_tokenize(only_words)
    removed_letters = [word for word in tokens if len(word)>2]
    lower_case = [l.lower() for l in removed_letters]
    #Removing stop words
    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    #Lammatizing
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

#Loading model and vocabulary
#RUnning model on unseen data

def main():
    #Loading the saved model
    model = joblib.load('svc.sav')
    #loading saved vocabulary
    vocabulary_model = pd.read_csv('vocabulary_SVC.csv', header=None)
    vocabulary_model_dict = {}
    for i, word in enumerate(vocabulary_model[0]):
         vocabulary_model_dict[word] = i
    tfidf = TfidfVectorizer(sublinear_tf=True, vocabulary = vocabulary_model_dict, min_df=5, norm='l2', ngram_range=(1,4)) # min_df=5 is clever way of feature engineering
  
    #Reading New reviews from dataset 
    Reviews_df = pd.read_csv('NoRatings.csv', encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', None) 
    # Using normalizer function on dataset 
    Reviews_df['normalized_reviwes'] = Reviews_df.Review.apply(normalizer)
    Reviews_df = Reviews_df[Reviews_df['normalized_reviwes'].map(len) > 0] # removing Null reviws rows
    print("Printing top 5 rows of dataframe showing original and cleaned Reviews....")
    print(Reviews_df[['Review','normalized_reviwes']].head())

    #Saving cleaned review to 'cleaned_review_new.csv'
    Reviews_df.to_csv('cleaned_review_new.csv', encoding='utf-8', index=False)
    #reading new dataset
    cleaned_Reviews_New = pd.read_csv("cleaned_review_new.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', None)
    
    #applying old model to new unseen data and predicting rating
    cleaned_Review_tfidf = tfidf.fit_transform(cleaned_Reviews_New['normalized_reviwes'])
    targets_pred = model.predict(cleaned_Review_tfidf)

    #Saving predicted rating to file
    cleaned_Reviews_New['predicted_rating'] = targets_pred.reshape(-1,1)
    cleaned_Reviews_New.to_csv('predicted_rating.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    main()

