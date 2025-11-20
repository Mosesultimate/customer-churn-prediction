# Customer Churn Prediction 🚀📊

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Dash](https://img.shields.io/badge/Dash-Plotly-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Welcome to the **Customer Churn Prediction** project!  
This repository provides a complete end-to-end solution for analyzing, predicting, and visualizing customer churn using **interactive dashboards**.

---

## **📁 Project Structure**

customer-churn-prediction/
├── data/                             # Cleaned dataset for modeling
├── models/                           # Saved/trained ML models
├── notebooks/                        # Jupyter notebooks (exploration, EDA, analysis)
├── src/                              # Feature engineering & preprocessing scripts
├── dashboards/                       # Interactive dashboards (Plotly Dash)
│   ├── churn_by_feature.py
│   ├── numeric_analysis.py
│   └── summary_cards.py
├── DATA_PREPROCESSING_ORDER_EXPLANATION.md  # Explanation of preprocessing steps
├── EDA_CODE_EXPLANATION.md                   # Explanation of EDA steps
├── app.py                             # Main Dash app to run dashboards
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation


---

## **✨ Features**

- **Feature Engineering:** Clean, scale, and transform raw data for modeling  
- **Multi-Dashboard Suite:**  
  - Churn by Categorical Feature  
  - Numeric Analysis (box plots, histograms)  
  - Summary Cards (churn rate, avg tenure, avg monthly charges)  
- **Interactive:** Dropdowns, tabs, hoverable charts  
- **Extensible:** Add new dashboards or preprocessing steps easily  

---

## 📸 Dashboard Screenshots**

> *(Screenshots of the dashboards)*

### Churn Overview Dashboard
![Churn Overview](./Images/Screenshot-2025-11-20-155808.png)

### Customer Insights Dashboard
![Customer Insights](./Images/Screenshot-2025-11-20-160018.png)

### Prediction Analysis Dashboard
![Prediction Analysis](./Images/Screenshot-2025-11-20-160123.png)

---

## **💻 Getting Started**

1. **Clone the repository**
```bash
git clone https://github.com/Mosesultimate/customer-churn-prediction.git
cd customer-churn-prediction

Install dependencies

pip install -r requirements.txt


Run the dashboard

python app.py


Open your browser at http://127.0.0.1:8050/ to view dashboards.
```

📈 Usage

Explore categorical churn patterns

Analyze numeric features distributions

View summary metrics for your customer base

Extend dashboards with new visualizations or filters

🛠️ Tech Stack

Python 3.x 🐍

Pandas & NumPy

Plotly & Dash

Scikit-learn

Git & GitHub

📄 License

MIT License © Moses Matola

💡 Author

Moses Matola
mosesmatola548@gmail.com
📊 GitHub Stats & Badges