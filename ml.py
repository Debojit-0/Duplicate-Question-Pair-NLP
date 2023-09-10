import pandas as pd
import numpy as np
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from string import punctuation
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


from sklearn.preprocessing import StandardScaler

import importlib
import sys

# Reload the sys module
importlib.reload(sys)
import pickle



new_df = pd.read_csv(r"C:\Users\DEBOJ\Downloads\Compressed\quora-question-pairs\C__Users_DEBOJ_Downloads_Compressed_quora-question-pairs_train1.csv")
#print(new_df)
new_df = new_df.dropna(axis=0)
new_df = new_df.sample(n=100000, random_state=42)
final_df =new_df.drop(columns=['id','qid1','qid2','question1','question2'])
print(final_df)
ques_df = new_df[['question1','question2']]
ques_df.head()
# merge texts
questions = list(ques_df['question1']) + list(ques_df['question2'])

#TFIDF VECTORISING

from sklearn.feature_extraction.text import TfidfVectorizer
# Create a TF-IDF vectorizer
# Create a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
# Transform the questions into TF-IDF matrices
tfidf_matrix = tfidf_vectorizer.fit_transform(questions)

# Split the TF-IDF matrix into two parts for 'question1' and 'question2'
q1_arr, q2_arr = np.vsplit(tfidf_matrix.toarray(), 2)

# Create DataFrames for the TF-IDF matrices
tfidf_df1 = pd.DataFrame(q1_arr, index=ques_df.index)
tfidf_df2 = pd.DataFrame(q2_arr, index=ques_df.index)

# Concatenate the DataFrames along the columns axis
tfidf_df = pd.concat([tfidf_df1, tfidf_df2], axis=1)

#Concatenate the DataFrames along the columns axis
final_df = pd.concat([final_df, tfidf_df], axis=1)

final_df.shape


#SPLIT INTO TRAIN TEST
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(final_df.iloc[:,1:].values,final_df.iloc[:,0].values,test_size=0.2,random_state=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
a = accuracy_score(y_test,y_pred)
print(a)

with open(r'C:\Users\DEBOJ\Desktop\quora duplicate\rf_model.pkl', 'wb') as file:
    pickle.dump(rf, file)

file_path = r'C:\Users\DEBOJ\Desktop\quora duplicate\tfidf_vectorizer.pkl'

# Save the tfidf_vectorizer as a pickle file
with open(file_path, 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)
