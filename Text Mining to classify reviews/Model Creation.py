
#Importing libraries
import re, nltk
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import joblib
from imblearn.over_sampling import SMOTE

#Normalising text to make it usefull
def normalizer(Review):
    soup = BeautifulSoup(Review)
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

#Using cross validation with kfold to find the best fitting model.
#Using Smote for oversampling data
def Cross_validation(data, targets, tfidf, clf_cv, model_name): #Performs cross-validation 

    kf = KFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
    scores=[]
    data_train_list = []
    targets_train_list = []
    data_test_list = []
    targets_test_list = []
    iteration = 0
    print("Performing cross-validation for {}...".format(model_name))
    for train_index, test_index in kf.split(data):
        iteration += 1
        print("Iteration ", iteration)
        data_train_cv, targets_train_cv = data[train_index], targets[train_index]
        data_test_cv, targets_test_cv = data[test_index], targets[test_index]
        data_train_list.append(data_train_cv) 
        data_test_list.append(data_test_cv) 
        targets_train_list.append(targets_train_cv) 
        targets_test_list.append(targets_test_cv)
        tfidf.fit(data_train_cv) # learning vocabulary of training set
        data_train_tfidf_cv = tfidf.transform(data_train_cv)
        #balancing Trainign dataset for each itteration 
        print("Number of observations in each class before oversampling (training data): \n", pd.Series(targets_train_cv).value_counts())
        smote = SMOTE(random_state = 101)
        data_train_tfidf_cv,targets_train_cv = smote.fit_sample(data_train_tfidf_cv,targets_train_cv)
        print("Number of observations in each class after oversampling (training data): \n", pd.Series(targets_train_cv).value_counts())
        #print shape of train and test data
        print("Shape of training data: ", data_train_tfidf_cv.shape)
        data_test_tfidf_cv = tfidf.transform(data_test_cv)
        print("Shape of test data: ", data_test_tfidf_cv.shape)
        clf_cv.fit(data_train_tfidf_cv, targets_train_cv) # Fitting model
        score = clf_cv.score(data_test_tfidf_cv, targets_test_cv) # Calculating accuracy
        scores.append(score) # appending cross-validation accuracy for each iteration
    print("List of cross-validation accuracies for {}: ".format(model_name), scores)
    mean_accuracy = np.mean(scores)
    print("Mean cross-validation accuracy for {}: ".format(model_name), mean_accuracy)
    print("Best cross-validation accuracy for {}: ".format(model_name), max(scores))

    #finding best cross-validation for best set
    max_acc_index = scores.index(max(scores)) #
    max_acc_data_train = data_train_list[max_acc_index]
    max_acc_data_test = data_test_list[max_acc_index]
    max_acc_targets_train = targets_train_list[max_acc_index] 
    max_acc_targets_test = targets_test_list[max_acc_index] 

    return mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test

#defing function for confusion matrixto find the result
def c_matrix(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, tfidf, targets, clf, model_name): #### Creates Confusion matrix for SVC
    tfidf.fit(max_acc_data_train)
    max_acc_data_train_tfidf = tfidf.transform(max_acc_data_train)
    max_acc_data_test_tfidf = tfidf.transform(max_acc_data_test)
    clf.fit(max_acc_data_train_tfidf, max_acc_targets_train) # Fitting SVC
    targets_pred = clf.predict(max_acc_data_test_tfidf) # Prediction on test data
    conf_mat = confusion_matrix(max_acc_targets_test, targets_pred)
    d={-1:'Negative', 0: 'Neutral', 1: 'Positive'}
    sentiment_df = targets.drop_duplicates().sort_values()
    sentiment_df= sentiment_df.apply(lambda x:d[x])
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=sentiment_df.values, yticklabels=sentiment_df.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix (Best Accuracy) - {}".format(model_name))
    plt.show()

#function to save SVC model and it's vocablary
def SVC_Save(data, targets, tfidf):
    tfidf.fit(data) # learn vocabulary of entire data
    data_tfidf = tfidf.transform(data)
    pd.DataFrame.from_dict(data=dict([word, i] for i, word in enumerate(tfidf.get_feature_names())), orient='index').to_csv('vocabulary_SVC.csv', header=False)
    print("Shape of tfidf matrix for saved SVC Model: ", data_tfidf.shape)
    clf = LinearSVC().fit(data_tfidf, targets)
    joblib.dump(clf, 'svc.sav')

#function to save NBC model and it's vocabulary
def NBC_Save(data, targets, tfidf):
    tfidf.fit(data) # learn vocabulary of entire data
    data_tfidf = tfidf.transform(data)
    pd.DataFrame.from_dict(data=dict([word, i] for i, word in enumerate(tfidf.get_feature_names())), orient='index').to_csv('vocabulary_NBC.csv', header=False)
    print("Shape of tfidf matrix for saved NBC Model: ", data_tfidf.shape)
    clf = MultinomialNB(alpha=1.0).fit(data_tfidf, targets)
    joblib.dump(clf, 'nbc.sav')


#COde to import data , clean data and run the model.
def main():
    #Reading training dataset as dataframe
    df = pd.read_csv("MedReviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', None) # Setting this so we can see the full content of cells
    # using normalizing fnction
    df['normalized_review'] = df.Review.apply(normalizer)
    df = df[df['normalized_review'].map(len) > 0] # removing rows with normalized tweets of length 0
    print("Printing top 5 rows of dataframe showing original and cleaned tweets....")
    print(df[['Review','normalized_review']].head())
    #droping unwanted column
    df.drop(['Medicine','Condition'], axis=1, inplace=True)
    #converting target column into numerical
    df['Rating']=df['Rating'].map({'High':1,'Low':0})
    #Saving cleaned Reviews in CSV
    df.to_csv('cleaned_data.csv', encoding='utf-8', index=False)
    # Reading cleaned Reviews from csv
    cleaned_data = pd.read_csv("cleaned_data.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', None)
    data = cleaned_data.normalized_review
    targets = cleaned_data.Rating

    #applying TFIDF function
    #We are using Trigrams and min_df= 20 
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=20, norm='l2', ngram_range=(1,4)) 

    # SVC Model
    SVC_clf = LinearSVC() 
    SVC_mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test = Cross_validation(data, targets, tfidf, SVC_clf, "SVC") # SVC cross-validation
    c_matrix(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, tfidf, targets, SVC_clf, "SVC") # SVC confusion matrix

    # NBC Model
    NBC_clf = MultinomialNB() 
    NBC_mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test = Cross_validation(data, targets, tfidf, NBC_clf, "NBC") # NBC cross-validation
    c_matrix(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, tfidf, targets, NBC_clf, "NBC") # NBC confusion matrix

    #saving the best model by comparing score
    if SVC_mean_accuracy > NBC_mean_accuracy:
        SVC_Save(data, targets, tfidf)
    else:
        NBC_Save(data, targets, tfidf)

if __name__ == "__main__":
    main()
