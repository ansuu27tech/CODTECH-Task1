# Customer Churn Analysis and Prediction Using Machine Learning

## 📝 Project Overview
This project is part of the **CODTECH Internship Task 3**. It presents a complete end-to-end Machine Learning pipeline to analyze customer data and predict whether a customer is likely to churn (leave the service) or not. It utilizes exploratory data analysis (EDA) for business insights and a Random Forest Classifier for robust prediction.

## 💼 Business Problem Statement
Customer retention is crucial for the profitability of any subscription-based or service-oriented business. Acquiring new customers is often significantly more expensive than retaining existing ones. The goal of this project is to build a predictive model that identifies customers at high risk of churning, allowing the business to proactively offer incentives, improve services, and enhance customer satisfaction before they leave.

## 📊 Dataset Description
The dataset contains customer demographic information, account details, and subscription services.
**Features include:**
- `Gender`, `Age`
- `Tenure` (Number of months the customer has stayed with the company)
- `MonthlyCharges`, `TotalCharges`
- `ContractType`, `InternetService`, `PaymentMethod`
- **Target Variable:** `Churn` (Yes/No)

## 💻 Technologies Used
- **Python:** Core programming language
- **Pandas & NumPy:** Data manipulation, cleaning, and mathematical operations
- **Scikit-learn:** Model building, Label Encoding, Random Forest Classifier, and performance metrics
- **Matplotlib & Seaborn:** Statistical data visualization

## ⚙️ Installation Guide
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ansuu27tech/CODTECH-Task3.git
   cd CODTECH-Task3
   ```
2. **Install Required Libraries:**
   Make sure you have Python installed. Run the following command to install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run
You can execute this project via the standalone script or the Jupyter Notebook:

**Option 1: Python Script**
```bash
python churn_analysis.py
```
This will run the full pipeline, print metrics to the console, and generate visualization images (`.png`) in the root directory.

**Option 2: Jupyter Notebook**
```bash
jupyter notebook churn_analysis.ipynb
```
Ideal for an interactive, step-by-step walkthrough of the data and visualizations.

## 📈 Results and Metrics
The Random Forest Classifier provides strong performance on tabular data with categorical and continuous variables. The evaluation outputs:
- **Accuracy:** Overall correctness of the model.
- **Precision:** Out of all predicted churns, how many actually churned?
- **Recall:** Out of all actual churns, how many were successfully identified?
- **F1 Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Breakdown of True Positives, True Negatives, False Positives, and False Negatives.

## 🖼️ Visualizations
The script generates the following key visualizations:
1. **Churn Distribution Chart** (Imbalance overview)
2. **Monthly Charges Analysis** (Distribution of charges relative to churn)
3. **Tenure Analysis** (Boxplot showing how tenure affects churn rate)
4. **Correlation Heatmap** (Feature relationships)
5. **Feature Importance Graph** (Which factors influence churn the most)

## 🔮 Future Enhancements
- Implement Hyperparameter Tuning (GridSearchCV/RandomizedSearchCV) to optimize the Random Forest model.
- Experiment with gradient boosting algorithms like XGBoost or LightGBM.
- Deploy the model via a REST API (FastAPI or Flask) or an interactive web app (Streamlit).

## 📸 Screenshots Section

**Correlation Heatmap:**
![Heatmap Placeholder](https://via.placeholder.com/600x400?text=Correlation+Heatmap+Placeholder)

**Feature Importance:**
![Feature Importance Placeholder](https://via.placeholder.com/600x400?text=Feature+Importance+Placeholder)

## 📁 Folder Structure
```text
CODTECH-Task3/
├── data/
│   └── customer_churn.csv        # Dataset
├── churn_analysis.py             # Main execution script
├── churn_analysis.ipynb          # Interactive notebook
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
└── LICENSE                       # MIT License
```
