# Email Spam Classifier Using Machine Learning

## 📝 Project Overview
This project is part of the **CODTECH Internship Task 2**. It is a complete Machine Learning pipeline for Natural Language Processing (NLP) designed to classify text messages or emails as **Spam** or **Ham** (Not Spam). The project demonstrates data cleaning, text preprocessing, feature extraction using TF-IDF, and classification using the Naive Bayes algorithm.

## 🎯 Objective
The main objective of this project is to build an accurate and efficient machine learning model that can automatically identify spam messages. This involves transforming raw text data into numerical features and training a classifier to predict the class of unseen text.

## 📊 Dataset Information
The dataset used in this project is a collection of labeled messages.
- **Features:** `text` (The raw message content)
- **Labels:** `label` ('spam' for unsolicited messages, 'ham' for normal messages)
- **Location:** `data/spam.csv`

## 💻 Technologies Used
- **Python:** Primary programming language.
- **Pandas & NumPy:** For data manipulation and numerical operations.
- **NLTK (Natural Language Toolkit):** For text preprocessing (tokenization, stopword removal).
- **Scikit-learn:** For machine learning (TF-IDF Vectorization, Naive Bayes, Metrics).
- **Matplotlib & Seaborn:** For data visualization and plotting the confusion matrix.

## ⚙️ Installation Guide
Follow these steps to set up the project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ansuu27tech/CODTECH-Task2.git
   cd CODTECH-Task2
   ```

2. **Ensure Python is installed:**
   Ensure you have Python 3.7+ installed on your machine.

3. **Install dependencies:**
   It is recommended to use a virtual environment. Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Execution Steps
You can run this project in two ways:

**Option 1: Run the Python Script**
Execute the main script to see the pipeline run in your terminal:
```bash
python spam_classifier.py
```
This will output the data summary, training progress, evaluation metrics, and generate a `confusion_matrix.png` file.

**Option 2: Run the Jupyter Notebook**
For an interactive, step-by-step experience with visualizations:
```bash
jupyter notebook spam_classifier.ipynb
```

## 📈 Results
The Multinomial Naive Bayes model performs exceptionally well on text classification tasks. 
- The model outputs an **Accuracy Score** demonstrating its overall correctness.
- The **Classification Report** provides detailed precision, recall, and F1-scores for both 'spam' and 'ham' classes.
- A **Confusion Matrix** is generated to visualize true positives, true negatives, false positives, and false negatives.

## 🔮 Future Improvements
- Implement advanced NLP techniques like Lemmatization or Stemming.
- Experiment with other algorithms (e.g., Support Vector Machines, Random Forest, or Deep Learning models like LSTM).
- Build a web interface (using Streamlit or Flask) to allow users to input text and get real-time spam predictions.

## 📸 Screenshots

**Terminal Output & Classification Report:**
![Classification Report](https://via.placeholder.com/800x400?text=Terminal+Output+Placeholder)

**Confusion Matrix:**
![Confusion Matrix](https://via.placeholder.com/600x400?text=Confusion+Matrix+Placeholder)

## 📁 GitHub Repository Structure

```text
CODTECH-Task2/
├── data/
│   └── spam.csv                  # Dataset containing spam/ham messages
├── spam_classifier.py            # Main Python script
├── spam_classifier.ipynb         # Interactive Jupyter Notebook
├── requirements.txt              # Project dependencies
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```
