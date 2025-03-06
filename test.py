import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
# load data
data  = pd.read_csv('creditcard.csv')
#seperate legitimate and fraudulent transactions 
legit = data[data.Class==0]
fraud = data[data.Class == 1]

#undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample,fraud],axis=0)

#split data into training and testing sets
X = data.drop('Class', axis=1)
Y = data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#train logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)
#evaluate model performance
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

#webapp
st.title("Credit Card Fraud detection model")
input_df = st.text_input('Enter All Required Features Values')
input_df_splited = input_df.split(',')

submit = st.button("Submit")

if submit:
    features = np.asarray(input_df_splited,dtype=np.float64)
    prediction = model.predict(features.reshape(1,-1))

    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent Transaction")
