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

#visualizations page
elif page=="Data Insights":

    st.title("Data Insights")

    df["date"]=pd.to_datetime(df["date"], dayfirst=True)
    df["Month"]=df["date"].dt.month
    df["Hour"]=df["date"].dt.hour

    tab1,tab2,tab3=st.tabs(["Time Analysis","Load Analysis","Correlations"])

    with tab1:
        st.subheader("Average Energy Usage per Month")
        fig,ax=plt.subplots(figsize=(10, 4))
        monthly_usage=df.groupby("Month")["Usage_kWh"].mean()
        ax.plot(monthly_usage.index, monthly_usage.values, marker='o',color="#3b82f6")
        ax.set_xlabel("Month")
        ax.set_ylabel("Usage (kWh)")
        ax.set_xticks(range(1,13))
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Average Energy Usage by Day of Week")
        day_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        fig, ax=plt.subplots(figsize=(10,4))
        day_usage=df.groupby("Day_of_week")["Usage_kWh"].mean().reindex(day_order)
        sns.barplot(x=day_usage.index,y=day_usage.values,ax=ax,palette="Set2")
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
        ax.set_ylabel("Average Usage (kWh)")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Energy Usage Heatmap (Hour vs Day)")
        pivot_table=df.pivot_table(values="Usage_kWh", index="Day_of_week", columns="Hour", aggfunc="mean")
        fig, ax=plt.subplots(figsize=(14, 5))
        sns.heatmap(pivot_table,cmap="YlOrRd", ax=ax)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        col1,col2=st.columns(2)
        with col1:
            st.subheader("Load Type Distribution")
            load_counts=df["Load_Type"].value_counts()
            fig,ax=plt.subplots(figsize=(6,6))
            ax.pie(load_counts,labels=load_counts.index,autopct='%1.1f%%')
            centre_circle=plt.Circle((0,0),0.70,fc='white')
            ax.add_artist(centre_circle)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Energy Usage Distribution by Load Type")
            fig, ax=plt.subplots(figsize=(10,5))
            sns.boxplot(x="Load_Type", y="Usage_kWh", data=df, palette="Set2", ax=ax)
            sns.stripplot(x="Load_Type", y="Usage_kWh", data=df, color="gray", alpha=0.2, ax=ax)
            ax.set_ylabel("Energy Usage (kWh)")
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Distribution of CO2 Emissions")
        fig,ax=plt.subplots(figsize=(10,4))
        sns.histplot(df["CO2(tCO2)"], bins=30,kde=True,ax=ax)
        ax.set_xlabel("CO2 Emissions (tCO2)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)
    with tab3:

        st.subheader("Power Factor vs Energy Usage")
        fig,ax=plt.subplots(figsize=(10,4))
        sns.scatterplot(x="Lagging_Current_Power_Factor", y="Usage_kWh", data=df, alpha=0.5, ax=ax)
        ax.set_xlabel("Lagging Current Power Factor")
        ax.set_ylabel("Energy Usage (kWh)")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Feature Correlation Matrix")
        fig, ax=plt.subplots(figsize=(10,6))
        corr=df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

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

#cost optimization page
elif page == "Cost Optimization":

    st.title("Cost Optimization")
    if "predicted_kwh" in st.session_state:
        predicted_kwh = st.session_state["predicted_kwh"]
        st.success(f"Using prediction: **{predicted_kwh:.2f} kWh**")
    else:
        st.warning("No prediction yet. Go to Predictions tab first or enter manually.")
        predicted_kwh = st.number_input("Enter kWh manually", min_value=0.0, value=10.0, step=0.1)

    st.subheader("Enter Your Electricity Rate")
    rate = st.number_input("Rate per kWh (Rs)", min_value=0.1, value=8.0, step=0.5)

    currentcost = predicted_kwh * rate

    st.subheader("Cost Breakdown")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted kWh",round(predicted_kwh,2))
    c2.metric("Rate per kWh", "Rs " +str(round(rate,2)))
    c3.metric("Estimated Cost","Rs "+str(round(currentcost,2)))

    st.subheader("Cost Comparison by Load Type")
    st.caption("What would it cost if the load type gets changed?")

    load_multipliers = {
        "Light_Load":0.4,
        "Medium_Load":0.75,
        "Maximum_Load":1.0
    }

    lc1, lc2, lc3 = st.columns(3)
    for col, (load, multiplier) in zip([lc1, lc2, lc3], load_multipliers.items()):
        cost=predicted_kwh*multiplier*rate
        col.metric(load.replace("_", " "), f"Rs {cost:.2f}")

    savings = (predicted_kwh*1.0*rate)-(predicted_kwh*0.4*rate)
    st.success(f"Switching from Maximum Load to Light Load saves **₹{savings:.2f}** per cycle!")

    st.subheader("Cost Projections")
    dailycost=currentcost*96  
    monthlycost=dailycost*30

    p1, p2 = st.columns(2)
    p1.metric("Estimated Daily Cost",f"Rs {dailycost:,.2f}")
    p2.metric("Estimated Monthly Cost",f"Rs {monthlycost:,.2f}")

#suggestions page
elif page=="Suggestions":
    st.title("Suggestions")

    predicted_kwh=st.session_state.get("predicted_kwh", None)
    load_type= st.session_state.get("load_type","Maximum_Load")
    hour= st.session_state.get("hour",14)
    week_type= st.session_state.get("week_type","Weekday")
    lagging_pf=st.session_state.get("lagging_pf",80)
    if predicted_kwh is None:
        st.warning("Run a prediction first to get personalized suggestions.")
        predicted_kwh =50.0

    co2=predicted_kwh*0.00042

    st.subheader("Current Energy Health")
    k1, k2, k3=st.columns(3)

    if predicted_kwh > 50:
        kwh_status="High"
    elif predicted_kwh > 20:
        kwh_status="Medium"
    else:
        kwh_status="Low"

    if co2 > 0.02:
        co2status="High"
    elif co2 > 0.008:
        co2status="Medium"
    else:
        co2status="Low"

    if 9<= hour<= 18:
        hourstatus="Peak"
    else:
        hourstatus="Off-Peak"

    k1.metric("kWh Usage",f"{predicted_kwh:.1f}",kwh_status)
    k2.metric("CO₂ (tCO₂)",f"{co2:.4f}",co2status)
    k3.metric("Operating Hour",f"{hour}:00",hourstatus)

    avg_kwh_overall=round(df["Usage_kWh"].mean(), 2)
    avg_kwh_by_load=df.groupby("Load_Type")["Usage_kWh"].mean()
    avg_for_load=round(avg_kwh_by_load.get(load_type,avg_kwh_overall), 2)

    st.subheader("Based on your inputs")

    if predicted_kwh>avg_for_load:
        st.warning(f"Your predicted usage of {round(predicted_kwh, 2)} kWh is "
                f"above the average of {avg_for_load} kWh for {load_type.replace('_', ' ')}. "
                f"Consider checking equipment efficiency.")
    else:
        st.success(f"Your predicted usage of {round(predicted_kwh, 2)} kWh is "
                f"below the average of {avg_for_load} kWh for {load_type.replace('_', ' ')}. "
                f"Good efficiency!")

    if lagging_pf < 85:
        st.warning(f"Your power factor is {lagging_pf} which is below 85. "
                f"This means energy is being wasted. Try to improve it.")
    else:
        st.success(f"Power factor is {lagging_pf} — that's good efficiency!")

    if 9 <= hour <= 18:
        avg_peak=round(df[df["Usage_kWh"].between(9, 18)]["Usage_kWh"].mean(), 2)
        avg_offpeak=round(df[~df["Usage_kWh"].between(9, 18)]["Usage_kWh"].mean(), 2)
        st.warning(f"Peak hours typically use more electricity. "
                f"Off-peak average is {avg_offpeak} kWh vs peak average of {avg_peak} kWh.")
    else:
        st.success("You are operating during off-peak hours. Good!")


    if co2 > 0.01:
        st.warning(f"CO₂ output is {round(co2, 4)} tCO₂. "
                f"Consider solar energy during daytime to bring this down.")

    st.subheader("General Tips")
    st.info("Keep power factor above 0.90 to avoid energy waste.")
    st.info("Try to run heavy machinery at night when demand is lower.")
    st.info("Spread out production tasks instead of running everything at once.")
    st.info("Switch to LED lighting to save on electricity bills.")
