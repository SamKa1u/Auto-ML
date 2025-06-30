import pandas as pd
import streamlit as st
import os

#   Profiling
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

#   ML
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import numpy as np

with st.sidebar:
    st.title('AutoML App')
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This app allows you to build a ML pipeline, using Streamlit and Ydata Profiling.")

if os.path.exists("sourceData.csv"):
    df = pd.read_csv("sourceData.csv", index_col=None)
    profile_report = ProfileReport(df, title="Data Profiling Report", explorative=True)

if choice == "Upload":
    st.title("Upload Your Data for Modeling")
    file = st.file_uploader(label="Upload Dataset")
    if file:
        df = pd.read_csv(file, index_col = None)
        df.to_csv("sourceData.csv", index = None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    st_profile_report(profile_report)


if choice == "ML":
    st.title("Machine Learning with PyTorch")

    # select target variable
    target = st.selectbox("Select Target Variable", df.columns)


    # select number of epochs
    epchs = st.slider("Number of Epochs", min_value=1, max_value=25, value=10, step=1)


    y = df[target]
    X_df = df.drop(columns=[target])

    # select features
    features = st.multiselect("Select columns NOT to be used for Feature Extraction", X_df.columns)
    X = X_df.drop(columns=features)

    # determine task type
    is_classification = True if y.dtype == "categorical" else False
    st.title(is_classification)
    st.info(y.dtype)

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
    class CustomModelA(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(X_train.shape[1], 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, len(torch.unique(y_train)) if is_classification else 1)
            )

        def forward(self, x):
            return self.net(x)


    # define models to compare
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    LinRegModel = LinearRegression()
    SupportVecRegModel = svm.SVR()
    DecTreeRegModel = DecisionTreeRegressor(max_depth=12)

    example_models = {
        "Linear Regression": LinRegModel,
        "Support Vector Regression": SupportVecRegModel,
        "Decision Tree Regressor": DecTreeRegModel,
    }

    for model_name, model in example_models.items():
        X_test_t = X_test.to(device)

        # train
        model.fit(X_train, y_train)
        # predict
        pred = model.predict(X_test_t)
        # evaluate
        score = model.score(X_test_t, y_test)

        if is_classification:
            pred = torch.argmax(torch.from_numpy(pred), dim=1).cpu().numpy()
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="weighted")
            results.append({
                "Model": model_name,
                "Accuracy": acc,
                "F1 Score": f1
            })
        else:
            results.append({
                "Model": model_name,
                "Score": score,
            })


    custom_models = {
        "Model A": CustomModelA(),
    }

    # setup info
    setup_info = {
        "Target Variable": target,
        "Task Type": "Classification" if is_classification else "Regression",
        "Feature Processing": "StandardScaler + One-Hot Encoding",
        "Validation Split": "80-20",
        "Batch Size": 32,
        "Epochs": epchs,
        "Optimizer": "Adam",
        "Loss Funtion": criterion.__class__.__name__,
    }

    st.info("The Experiment Settings")
    st.dataframe(pd.DataFrame([setup_info]))

    # training and evaluatiom

    for model_name, model in custom_models.items():
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # training loop
        model.train()
        for epoch in range(setup_info["Epochs"]):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # evaluation
        model.eval()
        with torch.no_grad():
            X_test_t = X_test.to(device)
            y_pred = model(X_test_t)
            if is_classification:
                y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                results.append({
                    "Model": model_name,
                    "Accuracy": acc,
                    "F1 Score": f1
                })
            else:
                y_pred = y_pred.flatten().cpu().numpy()
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results.append({
                    "Model": model_name,
                    "RMSE": rmse
                })

    # Display comparison
    results_df = pd.DataFrame(results)
    st.info("This is the ML Model Comparison")
    st.dataframe(results_df)




if choice == "Download":
    pass