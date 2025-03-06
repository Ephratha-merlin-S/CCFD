import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Loading the dataset to a Pandas DataFrame
credit_card_data =pd.read_csv('/content/credit_data.csv')

# Checking dataset information
credit_card_data.info()
print("Missing values:", credit_card_data.isnull().sum())
print("Class distribution:", credit_card_data['Class'].value_counts())

# Separating legitimate and fraudulent transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Balancing the dataset by sampling
legit_sample = legit.sample(n=len(fraud))
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Splitting features and labels
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Splitting into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train, Y_train)

# Evaluating Logistic Regression
log_train_pred = log_model.predict(X_train)
log_train_acc = accuracy_score(Y_train, log_train_pred)
log_test_pred = log_model.predict(X_test)
log_test_acc = accuracy_score(Y_test, log_test_pred)
print(f'Logistic Regression - Training Accuracy: {log_train_acc:.4f}, Test Accuracy: {log_test_acc:.4f}')

# Multi-Layer Perceptron Model
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam', random_state=2)
mlp_model.fit(X_train, Y_train)

# Evaluating MLP Model
mlp_train_pred = mlp_model.predict(X_train)
mlp_train_acc = accuracy_score(Y_train, mlp_train_pred)
mlp_test_pred = mlp_model.predict(X_test)
mlp_test_acc = accuracy_score(Y_test, mlp_test_pred)
print(f'MLP Classifier - Training Accuracy: {mlp_train_acc:.4f}, Test Accuracy: {mlp_test_acc:.4f}')
