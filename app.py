import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Energy Consumption Analyzer", layout="wide"
)

df=pd.read_csv("Steel_industry_data.csv")
randomforest=joblib.load("model2.pkl")
scaler=joblib.load("scaler.pkl")
encoderweekdays=joblib.load("encoder_day.pkl")
encoderload=joblib.load("encoder_load.pkl")
featureorder=joblib.load("feature_columns.pkl")
resultsmetrics=joblib.load("results.pkl").T

st.sidebar.title("Energy analyzer")
page=st.sidebar.radio("Go to",["Home","Data insights","Model comparison","Predictions","Cost optimization","Suggestions"])

#homepage
if page=="Home":
    st.title("Energy Consumption Analyzer")
    st.subheader("Dasboard for Steel Industry Energy")
    st.markdown(""" This dashboard analyses energy consumption in a steel manufacturing plant using Machine Learning. We have built and compared the models to predict electricity usage,estimate costs, and provide personalized suggestions.""" )
    st.subheader("Dataset Overview")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total records","35040")
    c2.metric("Data Year","2018")
    c3.metric("Recording Interval","Every 15 min")
    c4.metric("Industry","Steel plant")

    st.subheader("Model Summary")
    c1,c2,c3=st.columns(3)
    c1.metric("Models Trained","3")
    c2.metric("Best Model","Random Forest")
    c3.metric("Best R² score","0.9994")

    st.subheader("Tools & Technologies used:")
    c1,c2,c3,c4=st.columns(4)
    c1.info("Python")
    c2.info("Streamlit")
    c3.info("scikit-learn")
    c4.info("XGBoost")

    st.subheader("Project Objectives")
    st.write("Predict electricity consumption using ML models")
    st.write("Compare model performances using MAE,RMSE and R²")
    st.write("Estimate energy costs based on predictions")
    st.write("Provide suggestions to reduce energy waste")
    st.write("Deploy on cloud for real-world accessibility")

    st.caption("Use the sidebar to explore more!!")
