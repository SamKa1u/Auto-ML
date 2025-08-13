import pandas as pd
import streamlit as st
import os

#   Profiling
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

#   Fast API ML backend
import requests
import io
parquet_buffer = io.BytesIO()
url = 'http://127.0.0.1:8000'
df_endpoint = "/receive_df"

def process_df(df):
    df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    files = {'file':('data.parquet',parquet_buffer.getvalue(),'application/octet-stream')}
    response = requests.post(url+df_endpoint, files=files)
    return response

with st.sidebar:
    st.title('AutoML App')
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This app allows you to build a ML pipeline, using Ydata Profiling for EDA, a Streamlit frontend and a FastAPI backend.")

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
        res = process_df(df)
        if res["error"]:
            st.error(res["error"])
        else:
            st.info(res["message"])
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    try:
        st_profile_report(profile_report)
    except NameError:
        st.info("Upload Data before preceding to Profiling")

if choice == "ML":
    st.title("Machine Learning with Sci-kit Learn :robot:")
    if not df_exists:
        st.info("Upload Data before preceding to ML")
    else:
        st.title(":small[Experiment Settings]")
        # select target variable
        target = st.selectbox("Select Target Variable", df.columns)

        # select number of epochs
        epochs = st.slider("Number of Epochs", min_value=1, max_value=25, value=10, step=1)

        # select features
        features = st.multiselect("Select columns NOT to be used for Feature Extraction", X_df.columns)

        # select batch-size
        batch_size = st.slider("Train loader batch-size (in percent)", min_value=1, max_value=64, value=32, step=1)

        # select test_split
        test_split = st.slider("Percent of data reserved for validation", min_value=10, max_value=90, value=20, step=1)
        test_split = test_split / 100

        # get response from backend
        experiment_endpoint = f"/target/{target}/epochs/{epochs}/features/{features}/test_split/{test_split}/batch_size/{batch_size}"
        res = requests.post(url+experiment_endpoint)
        if res["error"]:
            st.error(res["error"])
        else:
            results = res["result"]
            setup = res["setup"]
            results_df = pd.read_json(results)
            setup_df =  pd.read_json(setup)

            # display experiment setup
            st.dataframe(setup_df)

            if is_classification:
                results = results_df
                for result in results:
                    st.subheader(f"{result['Model']}", divider="grey" )
                    st.title(":small[Classification Report:]")
                    reports_df =  pd.DataFrame.from_dict(result["ClsReps"]).transpose()
                    st.dataframe(reports_df)

                    # confusion matrix
                    CnfMtxs = result["CnfMtxs"]
                    st.info(CnfMtxs)
                    index = []
                    columns = []
                    for i in range(len(CnfMtxs)):
                        index.append(i)
                    for i in range(len(CnfMtxs[0])):
                        columns.append(i)
                    labels = ['Class 0', 'Class 1']
                    conf_matrix_df = pd.DataFrame(CnfMtxs, index=index, columns=columns)
                    conf_matrix_df.index.name = 'Actual'
                    conf_matrix_df.columns.name = 'Predicted'

                    fig = plt.figure(figsize=(4, 3))

                    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
                    plt.title('Confusion Matrix')
                    st.pyplot(fig)


            else:
                results_df = results_df.drop(columns="Predictions")
                st.title(":small[Model Comparison]")
                col1, col2 = st.columns(2)
                st.dataframe(results_df)
                fig = plt.figure(figsize=(4, 3))
                sns.lineplot(y_test)
                plt.title('Y_test')
                plt.xlim(0, len(y_test))
                plt.ylim(0, max(y_test))
                st.pyplot(fig)

                for result in results:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.lineplot(result["Predictions"], ax=ax)
                    plt.xlim(0,len(y_test))
                    plt.ylim(0,max(y_test))
                    ax.set_title(f'{result["Model"]}')
                    ax.set_xlabel('Index')
                    ax.set_ylabel('Y_pred')
                    st.pyplot(fig)


if choice == "Download":
    pass