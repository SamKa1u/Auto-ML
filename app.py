import pandas as pd
import streamlit as st
import os

from sympy.stats.crv_types import LogisticDistribution
#   Profiling
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

#   ML
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error, r2_score,median_absolute_error, classification_report, confusion_matrix
import numpy as np

def model_training(models, X_train, X_test, y_train, y_test,clas):
    preds = []
    R2s= []
    MAEs= []
    MSLEs= []
    MAPEs = []
    MAbEs = []
    ClsReps = []
    CnfMtxs = []
    for model_name, model in models.items():
        X_test_t = X_test.to(device)
        # train
        model.fit(X_train, y_train)
        # predict
        pred = model.predict(X_test_t)
        preds.append(pred)
        # evaluate
        if not clas:
            r2 = r2_score(y_test, pred)
            R2s.append(r2)

            mae = mean_absolute_error(y_test, pred)
            MAEs.append(mae)

            # st.info(y_test)
            # st.info(pred)
            msle = None#mean_squared_log_error(y_test, pred)
            MSLEs.append(msle)

            mape = mean_absolute_percentage_error(y_test, pred)
            MAPEs.append(mape)

            mabe = median_absolute_error(y_test, pred)
            MAbEs.append(mabe)
        else:
            cls_rep = classification_report(y_test, pred, output_dict=True)
            # cls_rep_df = pd.DataFrame(cls_rep, index=, columns=).transpose()
            # st.dataframe(cls_rep_df)
            ClsReps.append(cls_rep)

            con_mat = confusion_matrix(y_test, pred)
            CnfMtxs.append(con_mat)

    return preds, R2s, MAEs, MSLEs, MAPEs, MAbEs, ClsReps, CnfMtxs

with st.sidebar:
    st.title('AutoML App')
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This app allows you to build a ML pipeline, using Streamlit and Ydata Profiling.")

if os.path.exists("sourceData.csv"):
    df = pd.read_csv("sourceData.csv", index_col=None)
    profile_report = ProfileReport(df, title="Data Profiling Report", explorative=True)
    df_exists = True
else:
    df_exists = False


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
        st_profile_report(profile_report)
    except NameError:
        st.info("Upload Data before preceding to Profiling")

if choice == "ML":
    st.title("Machine Learning with PyTorch :robot:")
    if not df_exists:
        st.info("Upload Data before preceding to ML")
    else:
        st.title(":small[Experiment Settings]")
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
        is_classification = False if y.dtypes != "object" else True
        # st.title(is_classification)
        # st.info(y.dtypes)

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

        # define custom models
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

        svc = make_pipeline(StandardScaler(), svm.SVC(gamma="auto",kernel="rbf"))
        reg_models = {
            "Linear Regression": LinearRegression(),
            "Support Vector Regression": svm.SVR(),
            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=12),
        }
        clas_models = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Classifier": svc,
            "Naive Bayes": GaussianNB(),
        }
        example_models = clas_models if is_classification else reg_models

        # train example models
        preds, R2s, MAEs, MSLEs, MAbPEs, MAbEs, ClsReps, CnfMtxs = model_training(example_models, X_train, X_test, y_train, y_test, is_classification)
        i = 0
        for model_name, models in example_models.items():
            if not is_classification:
                results.append({
                    "Model": model_name,
                    "R2 Score": R2s[i],
                    "Mean absolute error": MAEs[i],
                    "Mean squared Log error": MSLEs[i],
                    "Mean absolute percentage error": MAbPEs[i],
                    "Median absolute error": MAbEs[i],
                })
            else:
                results.append({
                    "Model": model_name,
                    "ClsReps": ClsReps[i],
                    "CnfMtxs": CnfMtxs[i],
                })
            i += 1

        # compare custom models
        custom_models = {
            "Custom Model A": CustomModelA(),
        }

        # experiment setup info
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

        st.dataframe(pd.DataFrame([setup_info]))

        # custom model training and evaluation
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
                    ClsReps = classification_report(y_test, y_pred, output_dict=True)
                    CnfMtxs = confusion_matrix(y_test, y_pred)
                    results.append({
                        "Model": model_name,
                        "ClsReps": ClsReps,
                        "CnfMtxs": CnfMtxs,
                    })
                else:
                    y_pred = y_pred.flatten().cpu().numpy()

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    msle = mean_squared_log_error(y_test, y_pred)
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    mabe = median_absolute_error(y_test, y_pred)

                    results.append({
                        "Model": model_name,
                        "R2 Score": r2,
                        "Mean absolute error": mae,
                        "Mean squared Log error": msle,
                        "Mean absolute percentage error": mape,
                        "Median absolute error": mabe,
                    })

        # Display comparison
        results_df = pd.DataFrame(results)


        if is_classification:
            # st.header("Model Comparison", divider="grey")
            for result in results:
                cols = ["precision","recall","f1-score","support"]
                rows = [0,1, "accuracy"]
                st.subheader(f"{result['Model']}", divider="grey" )
                st.title(":small[Classification Report:]")
                reports_df =  pd.DataFrame.from_dict(result["ClsReps"]).transpose()
                st.dataframe(reports_df)

                # confusion matrix
                CnfMtxs = result["CnfMtxs"]
                labels = ['Class 0', 'Class 1']
                conf_matrix_df = pd.DataFrame(CnfMtxs, index=labels, columns=labels)
                conf_matrix_df.index.name = 'Actual'
                conf_matrix_df.columns.name = 'Predicted'

                fig = plt.figure(figsize=(4, 3))

                sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                st.pyplot(fig)


        else:
            st.title(":small[Model Comparison]")
            st.dataframe(results_df)





if choice == "Download":
    pass