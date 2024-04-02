#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from sklearn.metrics import accuracy_score


# In[ ]:


# Evaluation function for FNN
def evaluate_fnn(model, X_test, y_test):
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        outputs = model(X_test_tensor)
        predicted = torch.round(outputs)
        accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
        return accuracy


# In[ ]:


# Evaluation function for logistic regression
def evaluate_logistic_regression(model, X_test, y_test):
    """
    Evaluate a logistic regression model.

    Parameters:
    - model: Trained logistic regression model.
    - X_test (array-like): Features of the testing set.
    - y_test (array-like): Target of the testing set.

    Returns:
    - accuracy (float): Accuracy of the logistic regression model on the testing set.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# In[ ]:


# Evaluation function for SVM
def evaluate_svm(model, X_test, y_test):
    """
    Evaluate an SVM model.

    Parameters:
    - model: Trained SVM model.
    - X_test (array-like): Features of the testing set.
    - y_test (array-like): Target of the testing set.

    Returns:
    - accuracy (float): Accuracy of the SVM model on the testing set.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

