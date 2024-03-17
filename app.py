import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from pycaret.regression import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("https://cdn.pixabay.com/photo/2018/09/18/11/19/artificial-intelligence-3685928_1280.png")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application allow you build automated ML pipeline")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload you data!")
    file = st.file_uploader("Upload your dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)


if choice == "Profiling":
    if df is not None:
        st.title("Automated Exploratory Data Analysis")
        profile = ProfileReport(df, title="Pandas Profiling Report")
        with st.spinner("Generating Report....\nplease wait...."):
            st.components.v1.html(profile.to_html(), width=1000, height=1200, scrolling=True)


if choice == "ML":
    st.title("Machine Learning")
    target = st.selectbox("Select your target", df.columns)
    if st.button("Train model"):
        setup(df, target=target, verbose=False)
        setup_df = pull()
        st.info("This is the ML experiment")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML model")
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download":
    model_name = st.text_input("Name your model")

    if os.path.isfile("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            st.download_button("Download the Model", f, file_name=model_name + ".pkl")
    else:
        st.warning("No model available for download. Please ensure 'best_model.pkl' exists.")
