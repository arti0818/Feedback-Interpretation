import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv(r'C:\Users\LENOVO\Desktop\Full stack data science class\49. 30th Jan\30th\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
df1 = pd.concat([dataset,dataset], ignore_index=True)

# cleaning Text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


corpus = []

for i in range(0, 2000):
    review = re.sub('[^a-zA-Z]',' ', df1['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    #review = [ps.stem(word) for word in review if not word in set (stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Bag of word Model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = df1.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


classifiers = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVC':SVC(),
    'LogisticRegression':LogisticRegression(),
    'GaussianNB':GaussianNB(),
    'KNN':KNeighborsClassifier()
}

Comparision =pd.DataFrame(columns=['accuracy','bias','variance'])


# Loop through classifiers and print accuracy
for values, clf in classifiers.items():
    clf.fit(X_train, y_train) # fit the model
    y_pred= clf.predict(X_test) # then predict on the test set
    accuracy= accuracy_score(y_test, y_pred) # this gives us how often the algorithm predicted correctly
    variance = clf.score(X_test,y_test)
    bias=clf.score(X_train,y_train)
    #clf_report= classification_report(y_test, y_pred) # with the report, we have a bigger picture, with precision and recall for each class
    print(f"The accuracy of clf {type(clf).__name__} is {accuracy:.2f}")
    print(f"The variance of clf {type(clf).__name__} is {variance:.2f}")
    print(f"The bias of clf {type(clf).__name__} is {bias:.2f}")
   #----------------------------------------------------------
    #scores = cross_val_score(clf, X, y, cv=5)  # 5-fold cross-validation
    #print(f"{clf.__class__.__name__}: Accuracy = {scores.mean():.4f}") 
    Comparision.loc[values] = [accuracy, bias, variance]
  


