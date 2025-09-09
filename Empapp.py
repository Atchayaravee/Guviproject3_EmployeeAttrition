import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Load Data & Models
# ================================
@st.cache_data
def load_data():
    return pd.read_csv("Employee-Attrition-with-simulation.csv")  # includes simulation columns

@st.cache_resource
def load_models():
    try:
        attrition_model = joblib.load("model/attrition_model.pkl")
    except:
        attrition_model = None
    try:
        performance_model = joblib.load("model/performance_model.pkl")
    except:
        performance_model = None
    return attrition_model, performance_model

# Load everything
df_full = load_data()
attrition_model, performance_model = load_models()

# ================================
# Streamlit Layout
# ================================
st.set_page_config(page_title="Employee Attrition & Performance Dashboard", layout="wide")
st.title("📊 Employee Attrition & Performance Prediction Dashboard")
st.markdown("This dashboard provides **actionable HR insights** on attrition risk, performance ratings, and retention strategies.")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Attrition Prediction", "Performance Prediction", "HR Insights & Reports"]
)

# ================================
# 1. Overview
# ================================
if page == "Overview":
    st.subheader("Dataset Overview")
    st.write(df_full.head())
    st.metric("Total Employees", len(df_full))

    if "Attrition" in df_full.columns:
        predicted_attrition_rate = df_full["Attrition"].value_counts(normalize=True).get("Yes", 0)
        st.metric("Observed Attrition Rate", f"{predicted_attrition_rate*100:.2f}%")

        # Pie chart
        fig1, ax1 = plt.subplots()
        df_full["Attrition"].value_counts().plot.pie(
            autopct="%1.1f%%", startangle=90, ax=ax1,
            labels=["Stay", "Leave"], colors=["#66b3ff", "#ff6666"]
        )
        ax1.set_ylabel("")
        ax1.set_title("Attrition Distribution")
        st.pyplot(fig1)

        # Attrition by Department
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.countplot(data=df_full, x="Attrition", hue="Department", ax=ax2)
        ax2.set_title("Attrition Count by Department")
        st.pyplot(fig2)

# ================================
# 2. Attrition Prediction
# ================================
if page == "Attrition Prediction":
    st.header("🔮 Predict Employee Attrition")

    with st.form("attrition_form"):
        st.write("### Enter Employee Information")

        # Important categorical features
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        jobsat = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
        worklife = st.selectbox("Work-Life Balance (1=Bad, 4=Best)", [1, 2, 3, 4])
        envsat = st.selectbox("Environment Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
        bus_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        jobrole = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician", "Manager",
            "Manufacturing Director", "Healthcare Representative", "Human Resources", 
            "Sales Representative"
        ])
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
        ])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

        # Numeric inputs
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=30000, value=5000)
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3)

        # ✅ Move the submit button inside the form
        submitted = st.form_submit_button("Predict Attrition")

        if submitted:
            input_data = pd.DataFrame([{
                "OverTime": overtime,
                "JobSatisfaction": jobsat,
                "WorkLifeBalance": worklife,
                "EnvironmentSatisfaction": envsat,
                "BusinessTravel": bus_travel,
                "Department": dept,
                "JobRole": jobrole,
                "EducationField": education_field,
                "MaritalStatus": marital,
                "Age": age,
                "MonthlyIncome": monthly_income,
                "TotalWorkingYears": total_working_years,
                "YearsAtCompany": years_at_company,
            }])

            # One-hot encode and align with training features
            input_data = pd.get_dummies(input_data)
            if attrition_model is not None:
                input_data = input_data.reindex(columns=attrition_model.feature_names_in_, fill_value=0)
                prob = attrition_model.predict_proba(input_data)[0][1]
                st.metric("Attrition Probability", f"{prob:.2%}")
            else:
                st.error("Attrition model not found. Please train and save the model first.")


# ================================
# 3. Performance Prediction
# ================================
if page == "Performance Prediction":
    st.header("🌟 Predict Employee Performance Rating")

    with st.form("performance_form"):
        st.write("### Enter Employee Information")

        # Important categorical features
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        jobrole = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician", "Manager",
            "Manufacturing Director", "Healthcare Representative", "Human Resources", 
            "Sales Representative", "Research Director"
        ])
        dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
        ])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

        # Numeric inputs
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=30000, value=5000)
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
        engagement = st.selectbox("Engagement Score (1=Low, 4=High)", [1, 2, 3, 4])

        # Submit button inside form
        submitted = st.form_submit_button("Predict Performance Rating")

        if submitted:
            # Build raw input row
            perf_data = pd.DataFrame([{
                "OverTime": overtime,
                "JobRole": jobrole,
                "Department": dept,
                "EducationField": education_field,
                "MaritalStatus": marital,
                "Age": age,
                "MonthlyIncome": monthly_income,
                "TotalWorkingYears": total_working_years,
                "YearsAtCompany": years_at_company,
                "EngagementScore": engagement,
            }])

            # One-hot encode + align with training features
            perf_data = pd.get_dummies(perf_data)
            if performance_model is not None:
                perf_data = perf_data.reindex(columns=performance_model.feature_names_in_, fill_value=0)
                rating_num = performance_model.predict(perf_data)[0]

                # Map 0/1 to "Not High" / "High"
                rating_label = "High" if rating_num == 1 else "Not High"

                st.metric("Predicted Performance Rating", rating_label)
            else:
                st.error("Performance model not found. Please train and save the model first.")

# ================================
# 4. HR Insights & Reports
# ================================
elif page == "HR Insights & Reports":
    st.header("📌 HR Insights & Retention Strategies")

    # Feature Importance
    st.subheader("Top Features Influencing Attrition")
    importances = pd.Series(attrition_model.feature_importances_, index=attrition_model.feature_names_in_)
    top_feats = importances.sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    top_feats.plot(kind="barh", ax=ax)
    plt.gca().invert_yaxis()
    st.pyplot(fig)

    # Attrition by Job Role
    st.subheader("Attrition by Job Role")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(data=df_full, x="JobRole", hue="Attrition", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)


    # Retention Strategies
    st.subheader("Retention Strategy Simulation")

    # Define strategies
    strategies = [
        "Reduce overtime for employees currently working overtime",
        "Improve EngagementScore and JobSatisfaction by 1 point (max 4)",
        "Increase MonthlyIncome by 10% for employees earning below 5000",
        "Provide Stock Options level 2 for low-level employees"
    ]

    # Display strategies
    st.write("### Suggested Retention Strategies")
    for strat in strategies:
        st.markdown(f"- {strat}")

    # Check if simulation columns exist
    if "Attrition_Prob_After" in df_full.columns:
        avg_reduction = (df_full["Attrition_Prob_Original"] - df_full["Attrition_Prob_After"]).mean()
        st.metric("Average Reduction in Attrition Probability", f"{avg_reduction:.2%}")

        # Before vs After distribution plot
        fig, ax = plt.subplots(figsize=(7,5))
        sns.kdeplot(df_full["Attrition_Prob_Original"].dropna(), label="Before", shade=True, ax=ax)
        sns.kdeplot(df_full["Attrition_Prob_After"].dropna(), label="After", shade=True, ax=ax)
        ax.set_title("Attrition Probability Distribution (Before vs After Strategies)")
        ax.set_xlabel("Attrition Probability")
        plt.legend()
        st.pyplot(fig)

        # Top benefiting employees
        df_full["Reduction"] = df_full["Attrition_Prob_Original"] - df_full["Attrition_Prob_After"]
        top_benefit = df_full.sort_values(by="Reduction", ascending=False).head(10)
        st.write("### Top Employees Benefiting Most")
        st.dataframe(top_benefit[["EmployeeNumber","Attrition_Prob_Original","Attrition_Prob_After","Reduction"]].style.format({
            "Attrition_Prob_Original": "{:.2%}",
            "Attrition_Prob_After": "{:.2%}",
            "Reduction": "{:.2%}"
        }))
    else:
        st.info("Retention strategy simulation results not available in dataset. Please run simulation in training notebook.")
# ================================