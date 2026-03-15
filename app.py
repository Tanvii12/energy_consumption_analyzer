import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Energy Consumption Analyzer",
    layout="wide"
)

rnadomforest=joblib.load("model2.pkl")           
linearegression= joblib.load("model1.pkl")           
xgboost=joblib.load("model3.pkl")           
scaler=joblib.load("scaler.pkl")           
encoderdayforweekdays=joblib.load("encoder_day.pkl")      
encoderloadtype=joblib.load("encoder_load.pkl")    
featurecolumnsorder=joblib.load("feature_columns.pkl") 
results_df_metricscomparison=joblib.load("results.pkl").T     

df=pd.read_csv("Steel_industry_data.csv")
st.sidebar.title("Energy Analyzer")
page=st.sidebar.radio("Go to",["Home","Data Insights","Model Comparison","Predictions","Cost Optimization","Suggestions"])

#homepage
if page=="Home":

    st.title(" Energy Consumption Analyzer")
    st.subheader("Dashboard for Steel Industry Energy Management")

    st.markdown("""
    This dashboard analyzes energy consumption patterns in a steel manufacturing plant
    using Machine Learning. We have built and compared three models to predict electricity
    usage, estimate costs, and provide personalized suggestions.
    """)

    st.subheader("Dataset Overview")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Records","35,040")
    c2.metric("Data Year","2018")
    c3.metric("Recording Interval","Every 15 mins")
    c4.metric("Industry","Steel Plant")

    st.subheader("Model Summary")
    c1,c2,c3=st.columns(3)
    c1.metric("Models Trained","3")
    c2.metric("Best Model","Random Forest")
    c3.metric("Best R² Score","0.9994")

    st.subheader("Tools & Technologies used: ")
    c1,c2,c3,c4=st.columns(4)
    c1.info("Python")
    c2.info("Streamlit")
    c3.info("scikit-learn")
    c4.info("XGBoost")

    st.subheader("Project Objectives")
    st.write("Predict electricity consumption using ML models")
    st.write("Compare model performances using MAE, RMSE and R²")
    st.write("Estimate energy costs based on predictions")
    st.write("Provide suggestions to reduce energy waste")
    st.write("Deploy on cloud for real-world accessibility")

    st.caption("Use the sidebar to explore more!!")


#model comparison
elif page=="Model Comparison":

    st.title("Model Comparison")
    st.subheader("Performance Metrics")
    st.dataframe(results_df_metricscomparison, use_container_width=True)

    best_model=results_df_metricscomparison["r2_score"].idxmax()
    best_r2=results_df_metricscomparison.loc[best_model,"r2_score"]
    best_mae=results_df_metricscomparison.loc[best_model,"mae"]
    best_rmse=results_df_metricscomparison.loc[best_model,"rmse"]

    st.subheader("Best Model")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model",best_model)
    col2.metric("R² Score",round(best_r2,4))
    col3.metric("MAE (kWh)",round(best_mae,4))
    col4.metric("RMSE (kWh)",round(best_rmse,4))

    st.subheader("R² Score Comparison")
    model_names=results_df_metricscomparison.index.tolist()
    r2_vals=results_df_metricscomparison["r2_score"].tolist()

    fig,ax=plt.subplots(figsize=(8,8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    bars=ax.bar(model_names,r2_vals,color=["#ef4444", "#22c55e", "#3b82f6"], width=0.2)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + 0.002,str(round(bar.get_height(),4)),
                ha="center", va="bottom",fontsize=6,color="white")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("R² Score",color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)

#prediction page
elif page=="Predictions":

    st.title("Predictions")
    st.write("Model used: **Random Forest** — Best model (R² = 0.9994)")
    st.subheader("Enter Input Values")
    col1,col2=st.columns(2)
    with col1:
        lagging_reactive=st.slider("Lagging Current Reactive Power (kVarh)", 0.0, 200.0, 10.0)
        leading_reactive=st.slider("Leading Current Reactive Power (kVarh)", 0.0, 200.0, 5.0)
        lagging_pf=st.slider("Lagging Current Power Factor",0.0, 100.0, 80.0)
        leading_pf=st.slider("Leading Current Power Factor",0.0, 100.0, 90.0)
        nsm=st.slider("NSM (Seconds from Midnight)",0,86400, 3600, step=900)

    with col2:
        hour=st.selectbox("Hour of Day",list(range(0,24)))
        month=st.selectbox("Month",list(range(1,13)),format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
        day=st.selectbox("Day of Week",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        week_type=st.selectbox("Week Status",["Weekday","Weekend"])
        load_type=st.selectbox("Load Type",["Light_Load","Medium_Load","Maximum_Load"])

    if st.button("Predict Energy Consumption", use_container_width=True):

        day_enc=encoderdayforweekdays.transform([day])[0]
        load_enc=encoderloadtype.transform([load_type])[0]
        is_wkend=1 if week_type == "Weekend" else 0

        input_data=pd.DataFrame([[
            lagging_reactive,
            leading_reactive,
            lagging_pf,
            leading_pf,
            nsm,
            is_wkend,
            day_enc,
            load_enc,
            hour,
            month
        ]],columns=featurecolumnsorder)

        prediction = rnadomforest.predict(input_data)[0]

        st.session_state["predicted_kwh"]=prediction
        st.session_state["load_type"]=load_type
        st.session_state["hour"]=hour
        st.session_state["week_type"]=week_type
        st.session_state["lagging_pf"]=lagging_pf
        st.success(f"Predicted Energy Consumption: **{round(prediction,2)} kWh**")

        co2 = prediction * 0.00042
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted kWh",round(prediction,2))
        c2.metric("Estimated CO₂", str(round(co2,4))+" tCO₂")
        c3.metric("Load Type",load_type.replace("_", " "))

        st.info("Head to **Cost Optimization** tab to see the cost breakdown!!")

