# House Price Prediction Using Machine Learning

## CODTECH Internship Task 1

This project is part of the CODTECH Internship Task 1. It demonstrates the use of machine learning, specifically **Linear Regression**, to predict house prices based on various features like square footage, number of bedrooms, bathrooms, and the age of the house.

### 📁 Folder Structure

```text
codtech_task_1/
├── data/
│   └── house_data.csv             # Sample dataset
├── app.py                         # Python script for running the model
├── house_price_prediction.ipynb   # Jupyter Notebook with detailed analysis
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
```

### 🚀 Getting Started

#### Prerequisites

Ensure you have Python 3 installed. You can install the required dependencies using the `requirements.txt` file.

#### Installation & Execution

1. Clone the repository (or navigate to the project folder):
   ```bash
   git init
   git add .
   git commit -m "Initial commit for CODTECH Task 1"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Python script:
   ```bash
   python app.py
   ```

4. Alternatively, open the Jupyter Notebook to see step-by-step visualizations and model evaluation:
   ```bash
   jupyter notebook house_price_prediction.ipynb
   ```

### 🧠 Project Explanation

The project involves the following steps:
- **Data Loading:** The dataset is loaded using Pandas.
- **Data Cleaning:** Missing values are dropped to ensure a clean dataset for training.
- **Data Visualization:** Seaborn and Matplotlib are used to visualize feature distributions and correlations.
- **Train/Test Split:** The data is split into 80% training and 20% testing sets using Scikit-learn.
- **Model Training:** A Multiple Linear Regression model is trained on features `square_feet`, `bedrooms`, `bathrooms`, and `age_years`.
- **Evaluation:** The model is evaluated using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.

### 📸 Screenshots

*(Replace with actual screenshots of your output)*

**Data Visualization:**
![Visualization Placeholder](https://via.placeholder.com/600x400?text=Data+Visualization+Graph)

**Actual vs Predicted Prices:**
![Prediction Placeholder](https://via.placeholder.com/600x400?text=Actual+vs+Predicted+Graph)

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
