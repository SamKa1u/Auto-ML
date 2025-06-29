import pandas as pd
import streamlit as st
import os

#   Profiling
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

#   ML
import torch
import torh.nn as nn
import torch.optim as optim
from torh.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import numpy as np

with st.sidebar:
    st.title('AutoML App')
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This app allows you to build a ML pipeline, using Streamlit and Ydata Profiling.")

if os.path.exists("sourceData.csv"):
    df = pd.read_csv("sourceData.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Data for Modeling")
    file = st.file_uploader(label="Upload Dataset")
    if file:
        df = pd.read_csv(file, index_col = None)
        df.to_csv("sourceData.csv", index = None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    try:
        profile_report = ProfileReport(df, title="Data Profiling Report", explorative=True)
        st_profile_report(profile_report)
    except NameError:
        st.info("Upload a Dataset before preceding to Profiling.")

if choice == "ML":
    st.title("Machine Learning with PyTorch")

    # select target variable
    target = st.selectbox("Select Target Variable", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # determine task type
    is_classification = pd.api.types.is_categorical_dtype(y)

    # handle taret encoding if classification
    if is_classification:
        y = LabelEncoder().fit_transform(y)
        y = torch.tensor(y, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        metric_name = "Accuracy"
    else:
        y = torch.tensor(y.values, dtype=torch.float32)
        criterion = nn.MSELoss()
        metric_name = "MSE"

    # encode and scale features
    X = pd.get_dummies(X, drop_first = True)
    X = StandardScaler().fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # dataloader
    train_set = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    # define models
    class SimpleModelA(nn.Module):
        def __init__(self):
            super().__init__()
            self.net == nn.Sequential(
                nn.Linear(X_train.shape[1], 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, len(torch.unique(y_train)) if is_classification else 1)
            )

        def forward(self, x):
            return self.net(x)

    class SimpleModelB(nn.Module):
        def __init__(self):
            super().__init__()
            self.net == nn.Sequential(
                nn.Linear(X_train.shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, len(torch.unique(y_train)) if is_classification else 1)
            )

        def forward(self, x):
            return self.net(x)

    # define models to compare
    models = {
        SimpleModelA(),
        SimpleModelB()
    }

    # setup info
    setup_info = {
        "Target Variable": target,
        "Task Type": "Classification" if is_classification else "Regression",
        "Feature Processing": "StandardScaler + One-Hot Encoding",
        "Validation Split": "80-20",
        "Batch Size": 32,
        "Epochs": 10,
        "Optimizer": "Adam",
        "Loss Funtion": criterion.__class__.__name__,
    }

    st.info("The Experiment Settings")
    st.dataframe(pd.DataFrame([setup_info]))


if choice == "Download":
    pass