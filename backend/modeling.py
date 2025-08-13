# utils
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# preprocessing
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.metrics import mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error, r2_score,median_absolute_error, classification_report, confusion_matrix


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

            if np.any(pred < 0):
                msle = "MSLE not calculated due to negative predictions"
            else:
                msle = mean_squared_log_error(y_test, pred)
            MSLEs.append(msle)

            mape = mean_absolute_percentage_error(y_test, pred)
            MAPEs.append(mape)

            mabe = median_absolute_error(y_test, pred)
            MAbEs.append(mabe)
        else:
            cls_rep = classification_report(y_test, pred, output_dict=True)
            ClsReps.append(cls_rep)

            con_mat = confusion_matrix(y_test, pred)
            CnfMtxs.append(con_mat)
    return preds, R2s, MAEs, MSLEs, MAPEs, MAbEs, ClsReps, CnfMtxs

def get_modeling(target, epochs, df, features, test_split, batch_size):
    y = df[target]
    X_df = df.drop(columns=[target])

    # select features
    X = X_df.drop(columns=features)

    # determine task type
    is_classification = False if y.dtypes != "object" else True

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

    # dataloader
    train_set = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # define models to compare
    RanForReg = RandomForestRegressor(n_estimators=10, max_features=2, max_leaf_nodes=5, random_state=42)

    reg_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regression": RanForReg,
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=12),
    }
    clas_models = {
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
    }
    example_models = clas_models if is_classification else reg_models

    # experiment setup info
    setup_info = {
        "Target Variable": target,
        "Task Type": "Classification" if is_classification else "Regression",
        "Feature Processing": "StandardScaler + One-Hot Encoding",
        "Validation Split": "80-20",
        "Batch Size": 32,
    }

    # fit models
    preds, R2s, MAEs, MSLEs, MAbPEs, MAbEs, ClsReps, CnfMtxs = model_training(example_models, X_train, X_test, y_train, y_test, is_classification)

    # gather statistics
    results = []
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
                "Predictions": preds[i],
            })
        else:
            results.append({
                "Model": model_name,
                "ClsReps": ClsReps[i],
                "CnfMtxs": CnfMtxs[i],
            })
        i += 1

    results = pd.DataFrame(results).to_json()
    setup  = pd.DataFrame([setup_info]).to_json()
    return results, setup
