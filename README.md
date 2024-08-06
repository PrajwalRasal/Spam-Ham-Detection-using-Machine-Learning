# Email Spam Detection

This project aims to detect spam emails using a Naive Bayes classifier. The model is trained on a dataset of emails and their labels (spam or not spam) and can be used to classify new emails.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Setup and Installation](#setup-and-installation)
5. [Training the Model](#training-the-model)
6. [Evaluating the Model](#evaluating-the-model)
7. [Using the Model](#using-the-model)
8. [Results](#results)
9. [License](#license)

## Overview
The project involves:
- Loading and preprocessing the dataset.
- Creating a bag of words representation using `CountVectorizer`.
- Training a Naive Bayes classifier.
- Evaluating the classifier's performance.
- Classifying new emails using the trained model.

## Dataset
The dataset used for this project is an email dataset with two columns:
- `Category`: Indicates whether the email is 'spam' or 'ham'.
- `Message`: The content of the email.

## Model Architecture
The model is a Naive Bayes classifier, which is well-suited for text classification tasks. The emails are transformed into a bag of words representation using `CountVectorizer`.

## Setup and Installation
1. Install the required packages:
   ```bash
   pip install pandas numpy scikit-learn


Load the dataset:

import pandas as pd

df = pd.read_csv("F:\SE\SEM 2\PBL\spam - Email Dataset.csv")
df.head()

Preprocess the dataset:

df['spam'] = df['Category'].apply(lambda x: 1 if x =='spam' else 0)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.2)

Training the Model
Create a bag of words representation:

from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer()
X_train_cv = v.fit_transform(X_train.values)

Train the Naive Bayes classifier

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_cv, y_train)


Evaluating the Model
Transform the test data and make predictions:

X_test_cv = v.transform(X_test)
y_pred = model.predict(X_test_cv)

Evaluate the classifier's performance:

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

Using the Model
Create a pipeline for easier use:

from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)

Make predictions on new emails:
emails = [
    'NPTEL+: Upcoming Workshops - Hurry up and Register today!',
    'Summer Sale Is Live'
]

emails_count = v.transform(emails)
print(model.predict(emails_count))
