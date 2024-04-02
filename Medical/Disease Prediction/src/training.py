#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from src.models import FNN


# Train Logistic Regression:

# In[ ]:


def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.

    Parameters:
    - X_train (array-like): Features of the training set.
    - y_train (array-like): Target of the training set.

    Returns:
    - model: Trained logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


# Train SVM:

# In[ ]:


def train_svm(X_train, y_train, kernel='linear'):
    """
    Train an SVM model.

    Parameters:
    - X_train (array-like): Features of the training set.
    - y_train (array-like): Target of the training set.
    - kernel (str): Kernel function for SVM (default='linear').

    Returns:
    - model: Trained SVM model.
    """
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    return model


# Train FNN:

# In[ ]:


def train_fnn(X_train, y_train, input_size, num_epochs=10, batch_size=32):
    """
    Train a feedforward neural network model using PyTorch.

    Parameters:
    - X_train (array-like): Features of the training set.
    - y_train (array-like): Target of the training set.
    - input_size (int): Number of input features.
    - num_epochs (int): Number of training epochs (default=10).
    - batch_size (int): Batch size for training (default=32).

    Returns:
    - model: Trained feedforward neural network model.
    """
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Add a dimension for the single output

    # Instantiate the model
    model = FNN(input_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy loss for binary classification
    optimizer = optim.Adam(model.parameters())

    # Define DataLoader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


# 
