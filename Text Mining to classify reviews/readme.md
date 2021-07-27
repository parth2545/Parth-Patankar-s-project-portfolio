# Text Mining to Claddify Reviews

This project was a part of my accademic project.

Text Mining is a process of classifying unstructured text data into meaningful and 
actionable information. In our “MedReview.csv” dataset we have Reviews for different 
medicine for different condition. Reviews are classified into rating as “High” and “Low”. In 
our text mining technique we train our data on “ MedReviews” data and apply model on 
“NoRating” data to find out rating of reviews. We are going to clean the data , convert its 
TFIDF values ,balance the data to avoid underfitting or overfitting and then run Support Vector 
classifier model and naive Bayes classifier model to find the better model. In data preparation 
we have used trigrams as we saw many three words were taken in a row. This helped us 
improving bag of words.

**Support Vector classifier:** 

The SVC model simply find a best fitting hyperplane in N-dimensional space which 
distinctly classifies the data point. It finds a plane that has maximum margin. SVM looks at the 
feature’s interaction between each other and then classifies as two classes. After Kfold is 
performed dataset is divided in 9 train folds and 1 test fold for 10 iterations. Each set train on
SVC model and tries to find out best fitting hyperplane and its accuracy. After running the 
model, we found out that 7the iteration gives the best fold and accuracy of 
0.8888412017167382 i.e 88.88% . That means if 100 reviews are passed through this model it 
will be able to predict correct rating for 88 atleast. From the confusion matrix we can observe 
that our model only missed 227 reviews out of 2330. Which is quite good result.



**Naive Bayes classifier:**


NBC is probabilistic classification algorithm which consider each feature as 
independent component and try to find probability of it with respect to other component. Once
ran Naïve Bayes model and we discover that it only gives us accuracy of 0.8193908193908194 
i.e 81.90%. If looked at a confusion matrix, even if model is better in minimising false negative, 
it does not give good prediction. Around 363 reviews were wrongly classified while using NBC 
model.

Therefore, when we compare accuracy of both the model, we can directly say that 
support vector classifier performs much better at predicting the Rating. It gives us 8502 unique 
words as a vocabulary list with 8% more accuracy than another model. Hence, we save that 
model and run it on non-seen data(NoRating).

We cleaned the reviews, using the same normalizing function. Then we load our learned 
vocabulary and model. Once we verify the cleaned data, we ran our model on it. In that process 
our model gave TFIDF values using our vocabulary and try to classify it in two different plane. 
Our model does not understand which are high Rating and Low rating. All it does is, it assigns
0 to values in one hyperplane and 1 to values in others. When we went through the predicted 
rating, we found out 0 is classified as low rating as it has words like “horrible”, “Disgusting”,
“Not Good” etc. and the 1 as high cause it has words like “Nice”, “No side effect” , “perfect”
etc. Therefore, we can go ahead and say we successfully classified Rating into High and Low. 
If we assume that our model was able to give us same 88% accuracy(which is not ideal), we 
can say that out of 852 reviews it must have at least predicted 750 correct ratings.

