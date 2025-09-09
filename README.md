# Employee Attrition & Performance Prediction Dashboard  

📌 **Project Overview**  

This project provides HR teams with a **machine learning-powered dashboard** to analyze and predict employee attrition and performance. It helps organizations **identify at-risk employees**, forecast **performance ratings**, and design **retention strategies** using an interactive **Streamlit app**.  

---

## 🔑 The system integrates:  
- **Exploratory Data Analysis (EDA):** Understanding employee attrition patterns  
- **Attrition Prediction Model:** Predict probability of employees leaving  
- **Performance Prediction Model:** Classify employees as *High* or *Not High* performers  
- **Retention Strategy Simulation:** Evaluate HR interventions (salary increase, overtime reduction, engagement boost, stock options)  
- **Interactive Dashboard:** Built with **Streamlit** for real-time HR insights  

---

## ⚙️ Tech Stack  

- **Python**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Streamlit**: Interactive dashboards  
- **Joblib/Pickle**: Model serialization  
- **Jupyter Notebook**: Data preprocessing, model training  

---

## 📂 Project Structure  
Employee_Attrition_Project/
│── data/
│ ├── Employee-Attrition(raw).csv # Raw dataset
│ ├── Employee-Attrition(cleaned).csv # Cleaned dataset
│ ├── Employee-Attrition-with-simulation.csv # After retention strategies
│
│── model/
│ ├── attrition_model.pkl # Trained attrition prediction model
│ ├── performance_model.pkl # Trained performance prediction model
│
│── Empmodfinal.ipynb # Notebook for EDA, training & simulation
│── Empapp.py # Streamlit dashboard
│── README.md # Project documentation

---

## 🚀 Features  

✅ **Attrition Trends:** Attrition distribution by department, job role, and demographics  
✅ **Attrition Prediction:** Estimate employee attrition probability  
✅ **Performance Prediction:** Predict performance rating (*High* / *Not High*)  
✅ **Retention Strategies:** Test HR actions like salary hikes, overtime reduction, stock options  
✅ **HR Insights:** Feature importance, top employees benefiting from interventions  
✅ **Downloadable Reports:** Export data for HR teams  

---

## 📊 Dashboard Preview  

- **Home / Overview:** Dataset stats, attrition pie chart, attrition by department  
- **Attrition Prediction:** Predict attrition probability from employee details  
- **Performance Prediction:** Classify employees as *High* or *Not High* performers  
- **HR Insights:** Key drivers of attrition, impact of retention strategies  

---

## 🧠 Model Evaluation  

### Attrition Model  
- Algorithm: **Random Forest Classifier** (best performing)  
- Accuracy: ~87%  
- Key features: OverTime, JobSatisfaction, MonthlyIncome, YearsAtCompany, Age  

### Performance Model  
- Algorithm: **Random Forest Classifier**  
- Accuracy: ~82%  
- Target classes: *High* vs *Not High*  

---

## ▶️ How to Run the Dashboard  

1. **Clone this repo**  
```bash
git clone https://github.com/yourusername/Employee_Attrition_Project.git
cd Employee_Attrition_Project
Create a virtual environment and install dependencies:

* pip install -r requirements.txt

* Run Streamlit app:

* streamlit run app.py


Open the link in your browser (default: http://localhost:8501)
📥 HR Reports

The dashboard generates simulation-based reports that help HR teams:

Estimate attrition risk before & after retention strategies

Identify top employees who benefit most from HR actions

📌 Business Value

✅ Reduce turnover costs with proactive retention
✅ Forecast workforce performance & productivity
✅ Data-driven HR decision making
✅ Strengthen engagement & employee satisfaction

📜 License

This project is licensed under the MIT License – feel free to use, modify, and share with attribution.

🔥 With this dashboard, HR teams can turn employee data into actionable insights for better retention and workforce planning!


