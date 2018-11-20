#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 23:15:06 2018

@author: luki
"""

import numpy as np 
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#import data
train= pd.read_csv('../database/train.csv')
test = pd.read_csv('../database/test.csv')
submission = pd.read_csv('../database/sample_submission.csv')
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
#Data cleaning
def clean_text(comment_text):
    comment_list = []
    for text in comment_text:
        # Conver words to lowercase
        text = text.lower()
        # Delete non-alphanumeric characters
        text = re.sub(r"[^A-Za-z0-9(),!?@&$\'\`\"\_\n]", " ", text)
        text = re.sub(r"\n", " ", text) 
        # restore abbreviation
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        # replace some special symbols
        text = text.replace('&', ' and')
        text = text.replace('@', ' at')
        text = text.replace('$', ' dollar')   
        comment_list.append(text)
    return comment_list
train["clean_comment_text"] = clean_text(train['comment_text'])
test['clean_comment_text'] = clean_text(test['comment_text'])
all_comment_list = list(train['clean_comment_text']) + list(test['clean_comment_text'])
# TF-IDF building
text_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',token_pattern=r'\w{1,}',
                         max_features=20000, ngram_range=(1, 1), analyzer='word')
text_vector.fit(all_comment_list)
train_vec = text_vector.transform(train['clean_comment_text'])
test_vec = text_vector.transform(test['clean_comment_text'])
# separate training dataset (80%) and testing dataset (20%)
x_train, x_valid, y_train, y_valid = train_test_split(train_vec, train[labels], test_size=0.2, random_state=2018)
x_test = test_vec

# Use Logistic Regression for each and get confusion matrix for each label
accuracy = []
for label in labels:
    lr = LogisticRegression(C=6)
    lr.fit(x_train, y_train[label])
    y_pre = lr.predict(x_valid)
    #Confusion matrix
    cm = confusion_matrix(y_valid[label], y_pre)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['non-'+label,label]
    plt.title('Confusion Matrix---'+label)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(cm[i][j]))
    plt.show()
    # print CV score
    print(cross_val_score(lr, x_train, y_train[label], cv=3, scoring='roc_auc'))
    cv_score = np.mean(cross_val_score(lr, x_train, y_train[label], cv=3, scoring='roc_auc')) 
    valid_scores = accuracy_score(y_pre, y_valid[label])
    #print average score of CV
    print("{} train cv score is {}, valid score is {}".format(label, cv_score, valid_scores))
    accuracy.append(valid_scores)
    pred_proba = lr.predict_proba(x_test)[:, 1]
    submission[label] = pred_proba
# print total accuracy
print("Total  accuracy is {}".format(np.mean(accuracy)))
# sumbmission of test file
submission.to_csv('../result/submission_LR.csv', index=False)