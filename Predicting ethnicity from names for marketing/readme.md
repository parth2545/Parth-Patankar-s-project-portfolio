# Predicting Ethnicity from Names for Marketing

- #### Abstract

  Text categorization has seen a variety of development in recent years. Deriving ethnicity, 
gender, and race information from names have many marketing, social science, & medical 
applications. Here we present the report on classifying the names into five ethnic groups for 
application in marketing. We exploit data of around 600K names from different 
sources and attempt to find a relationship between names and ethnicity using machine learning 
models such as SVM, NBC, and LSTM. We used various types of features and techniques to 
find out the best-performing model. Our models effectively performed on the test data, giving 
a very good accuracy and precision and recall score. We presented the application of this 
classifier on real-life data of the company and demonstrated how it would minimize the manual 
work, errors, and expenses.


- #### Data collection and Data preperation

  Data plays a vital role in machine learning models. It is proved that the more the data, 
the better it will be as it plays the role of vocabulary in natural language processing models.Having personal data with ethnicity information is not 
widely available. For this research, we had to extract the data from different sources and then 
combine them in one and label them. We gathered data from Florida for voter’s registration, wikipedia catagorical data extracted by ambekar, Indian and pakistan's general election candidate data and also extracted babies name from momjunction.com and many small sources.For final data was like following : 

 Ethnicity | Number of Records
  ------------ | -------------
 Asian – Indian Subcontinent | 130488
 Asian – East Asian | 16121
 Hispanic | 60953
 White non-Hispanic| 422692
 Black non-Hispanic |45684
 Total | 675938
 
 For the implementation of the model, We used real-life data; the company has provided 
their 100k people data including past manually extracted data to verify the results.

- #### Models created to find the best fitting model

  Text categorization is a process of sorting text documents into given classes. This 
classification has seen huge progress since the deep learning model, and text conversion 
technology have improved. This categorization is mainly based on human judgment, keywords 
or features clustering or learning algorithms. To establish a relationship between a name and 
given ethnicity, we used three different models with different parameters.Using a complete name as vocabulary puts limits on the prediction, and the model 
cannot classify the names which are out of the vocabulary list. The characters 
in the name can play a significant role in feature selection and predictions. The distance between the characters can also be used to find the relations between 
two entities. We make use of Support vector classification (SVC), Naive Bayes classification
(NBC), and LSTM (long-short term memory) model. We build model using different features 
and parameters, Models are as follows:
  1) SVM – First name – Bi-gram/Tri-gram feature
  2) [SVM – Full name – Name as feature](https://github.com/parth2545/Parth-Patankar-s-project-portfolio/blob/main/Predicting%20ethnicity%20from%20names%20for%20marketing/SVM_model.ipynb) 
  3) SVM – Full name – Bi-gram/ Tri-gram Feature
  4) NBC – First Name – Bi-gram/Tri-gram feature
  5) [NBC - Full name – Name as feature](https://github.com/parth2545/Parth-Patankar-s-project-portfolio/blob/main/Predicting%20ethnicity%20from%20names%20for%20marketing/NBC_Name_model.ipynb)
  6) NBC - Full name – Bi-gram/ Tri-gram Feature
  7) [LSTM - Full name – Single alphabet](https://github.com/parth2545/Parth-Patankar-s-project-portfolio/blob/main/Predicting%20ethnicity%20from%20names%20for%20marketing/LSTM_Single_alphabet.ipynb)
  8) LSTM – Full name – Name as token 
  9) LSTM – Full name – Bi-gram 

  We prepared models using such combination, they were easy to compared to find the most 
accurate model. Low accurate model was disqualified as it might wrongly classifying potential 
customer which would be loss of customer for a company. 

  We used vectorisation/ n-gram feature selection, SMOTE- Oversampling, cross- validation techniques to avoid problems like overfitting, reducing biases and measure model's accuracy and skill.

  The final and most accurate models are LSTM - Full name - Single alphabet , SVM - Full name , NBC - Full Name respectively.
  
- #### Results
    Interpreting result is the most important task of any project. Outcomes from the 
models need to be correctly interpreted to find out the best performing model. In machine 
learning, even though , high scores does not truly mean model is correct. We must compare 
the result to each other to find out final model


 Model | Precision | Recall | F1- score | Accuracy | Vocabulary count
------------ | ------------- | ------------ | ------------- | ------------ | ------------ 
  SVM  Full Name | 0.93 | 0.98 | 0.96 | 0.90 | 790302
  NBC Full Name | 0.89 | 0.88 | 0.85 | 0.88 | 790302
  LSTM – single | 0.90 | 0.91 | 0.90 | 0.91 | 26


  We chose to focus on recall score to find best performing model. We do 
not want our model to miss any false negative as its better for the company. Therefore, this 
model gives us more false-positive over saving some false negative. That means, this model 
will give a Hispanic person as an Asian- Indian subcontinent person than an Asian – Indian 
Subcontinent origin person as Hispanic. Therefore we chose models with highest recall scores. We got an accuracy of 0.91 in the LSTM model along 
with 0.90 and 0.88 in the Support vector classifier and a Naive Bayes model. We implemented 
all these three models on the company's unseen data and concluded that LSTM and SVM are 
the best working model in the real world where the naïve Bayes model creates many false 
negatives. 

- #### Application in marketing
  Multi culture marketing is the practice of marketing to one or more audiences of a specific ethnicity. The marketing campaign directed toward a specific group of people tends to give better results than generic campaigns. This type of marketing takes advantage of cultural references, ethnic traits, concepts, and traditions. Our models were used for a US company to find Asian- Indian people. As of now, the company works with four county data. The average data per county is around 50000. Currently, they manually extract people who are of "Asian – Indian Subcontinent" origin. The system is very time-consuming, inefficient, and costs a company many human resource hours. The company's central vision is to expand its operation. That is where we are going to use our name – ethnicity classifier. This classifier will allow the company to save around 70-80% of its human resource and atomize the complete procedure.
