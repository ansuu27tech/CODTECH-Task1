# Movie Recommendation Engine Using Machine Learning

## 📝 Project Overview
This project is part of the **CODTECH Internship Task 4**. It demonstrates how to build a Content-Based Filtering recommendation system from scratch using Machine Learning. The engine recommends movies to users based on similarities in movie attributes, specifically utilizing text descriptions, genres, and keywords.

## 🎯 Objective
The goal is to develop an algorithm that can understand the contextual similarity between different movies. By inputting a movie title, the engine processes its associated text features and outputs the Top 10 most similar movies, enabling a personalized discovery experience for users.

## 📊 Dataset Description
The dataset contains a subset of popular movies with their textual metadata.
**Dataset Columns:**
- `title`: The name of the movie
- `genre`: Categories the movie belongs to (e.g., Action, Sci-Fi, Drama)
- `overview`: A brief summary of the movie's plot
- `keywords`: Key themes and tags associated with the movie
- **Location:** `data/movies.csv`

## 💻 Technologies Used
- **Python:** The main programming language.
- **Pandas & NumPy:** For data loading, manipulation, and numerical arrays.
- **Scikit-learn:** For Natural Language Processing (`TfidfVectorizer`) and mathematical metric computations (`cosine_similarity`).
- **Matplotlib:** For visualizing dataset statistics and similarity scores.

## ⚙️ Installation Instructions
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ansuu27tech/CODTECH-Task4.git
   cd CODTECH-Task4
   ```
2. **Install Dependencies:**
   Ensure Python 3 is installed. Run the following command:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage Guide
You can execute this recommendation engine in two formats:

**Option 1: Python Script**
Run the main script to see terminal outputs and generate visualization PNGs:
```bash
python movie_recommender.py
```

**Option 2: Jupyter Notebook**
For an interactive experience where you can dynamically change the input movie:
```bash
jupyter notebook movie_recommender.ipynb
```

## 🧠 Recommendation Workflow
1. **Read Dataset:** Loads the movie dataset into a Pandas DataFrame.
2. **Preprocess Text:** Cleans missing values (NaN to empty strings) and concatenates `genre`, `overview`, and `keywords` into a single text document (`combined_features`).
3. **Create Feature Vectors:** Applies `TfidfVectorizer` to convert text into numerical vectors, assigning importance to rare words and penalizing common ones.
4. **Compute Similarity Matrix:** Calculates the `cosine_similarity` between all movies based on their TF-IDF vectors, resulting in a matrix where `1.0` means identical and `0.0` means entirely dissimilar.
5. **Accept Movie Input:** Takes a search query (e.g., "The Matrix").
6. **Return Top 10:** Sorts the similarity scores in descending order and returns the titles of the top 10 most similar movies.

## 📋 Example Output
When searching for **"The Matrix"**:
```text
Top 10 Recommendations for 'The Matrix':
1. Inception
2. The Avengers
3. Avatar
4. Star Wars: Episode IV - A New Hope
5. Interstellar
...
```

## 📸 Screenshots Section

**Genre Distribution:**
![Genre Distribution](https://via.placeholder.com/600x400?text=Genre+Distribution+Chart)

**Similarity Statistics:**
![Similarity Statistics](https://via.placeholder.com/600x400?text=Similarity+Distribution+Chart)

## 🔮 Future Enhancements
- Switch from Content-Based Filtering to Collaborative Filtering using user ratings (e.g., Matrix Factorization or SVD).
- Implement a Hybrid Recommender System combining Content and Collaborative approaches.
- Deploy the model online using a web framework like **Streamlit** or **Flask**.

## 📁 Folder Structure
```text
CODTECH-Task4/
├── data/
│   └── movies.csv                 # Movie metadata
├── movie_recommender.py           # Main engine script
├── movie_recommender.ipynb        # Interactive notebook
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
└── LICENSE                        # MIT License
```
