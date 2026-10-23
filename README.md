# Sentiment Analysis Using Machine Learning

## 📌 Project Overview

This project focuses on analyzing text reviews and classifying their sentiment into three categories: **Positive, Negative, and Neutral** using Machine Learning and Natural Language Processing (NLP).

The project includes data analysis, text preprocessing, model training, evaluation, and an interactive web application built with Streamlit.

## 🎯 Objectives

- Analyze textual reviews and identify their sentiment.
- Perform data cleaning and exploratory data analysis (EDA).
- Preprocess text data using NLP techniques.
- Train and evaluate machine learning models.
- Predict sentiment and display class probabilities.
- Build an interactive application for user-provided reviews.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas & NumPy** — Data processing
- **Matplotlib & Seaborn** — Data visualization
- **NLTK & spaCy** — Natural Language Processing
- **Scikit-learn** — Machine learning and model evaluation
- **TextBlob & WordCloud** — Text analysis and visualization
- **Jupyter Notebook** — Data analysis and model development
- **Streamlit** — Interactive web application
- **Joblib** — Saving and loading the trained model

## 📂 Project Structure

```text
Sentiment Analysis/
│
├── app.py
├── requirements.txt
├── dataset.xlsx
├── sentiment_analysis_eda.xlsx
├── Sentiment_Analysis.ipynb
├── sentiment_model.pkl
├── README.md
└── .gitignore
```

**Note:** `.ipynb_checkpoints/` and `__pycache__/` are excluded from Git using `.gitignore`.

## 📊 Dataset

The project uses an Excel dataset containing text reviews and associated information for sentiment analysis.

The notebook explores the dataset, performs preprocessing, and prepares data for model development.

If sentiment labels are derived from ratings, they should be understood as proxy labels rather than independently verified human sentiment annotations.

**Dataset file:** `dataset.xlsx`

## ⚙️ Project Workflow

1. Load and inspect the dataset.
2. Clean and preprocess the text data.
3. Perform exploratory data analysis.
4. Prepare sentiment labels and training data.
5. Train and compare machine learning models.
6. Evaluate model performance using classification metrics.
7. Save the selected model using Joblib.
8. Load the trained model in the Streamlit application.
9. Enter a review and obtain a predicted sentiment.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "Sentiment Analysis"
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a compatible Python and scikit-learn version for the saved model.

If the application requires the spaCy English language model, install it using:

```bash
python -m spacy download en_core_web_sm
```

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL where you can interact with the application.

## 💻 Application Features

- Text input for user reviews
- Automatic text preprocessing
- Positive, Negative, or Neutral sentiment prediction
- Probability scores for supported sentiment classes
- Interactive interface built with Streamlit

## 📈 Model Evaluation

The Jupyter Notebook contains the project's data analysis, model development, and evaluation workflow.

Performance should be assessed using appropriate metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Macro F1-score is useful when comparing performance across all three sentiment classes, particularly when class distributions are imbalanced.

Actual performance depends on the dataset, model, and evaluation results.

## 🔮 Future Improvements

- Improve performance on Neutral reviews.
- Evaluate additional machine learning models.
- Test on a separate, independently labeled dataset.
- Improve text preprocessing and handling of different writing styles.
- Deploy the Streamlit application online.

## 👨‍💻 Author

**Mihir Prajapati**

Interested in Data Science, Machine Learning, Artificial Intelligence, and Natural Language Processing.

## 📄 License

A license has not yet been specified. Add an appropriate license if you intend to distribute or reuse this project publicly.
