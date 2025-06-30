# RestoByte: Restaurant Review Sentiment Analysis App

RestoByte is a sentiment analysis web application built using Python, Streamlit, and machine learning techniques. It processes restaurant reviews and classifies them into Positive, Negative, or Neutral sentiments. The application provides meaningful insights through interactive visualizations and allows users to enter their own reviews for real-time sentiment analysis.

---

## Project Overview

This project demonstrates the use of Natural Language Processing (NLP) and machine learning to analyze customer sentiment in the restaurant domain.

### Key Features

- Sentiment classification using a Naive Bayes classifier enhanced with keyword-based refinement
- Real-time prediction for user-submitted reviews
- Sentiment distribution visualized through pie and bar charts
- Word cloud generation for each sentiment category
- Downloadable CSV of sentiment-labeled review data
- Clean, responsive user interface built using Streamlit

---

## Technologies Used

- **Programming Language**: Python
- **Frontend / UI**: Streamlit
- **Libraries**:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - wordcloud
  - nltk
  - scikit-learn

- **Model Used**: Multinomial Naive Bayes
- **Text Processing**: NLTK (Lemmatization, Stopword Removal)
- **IDE Used**: Visual Studio Code (VS Code)
- **Deployment**: Streamlit Cloud

---

## Dataset

- **Source**: Kaggle
- **File**: `Restaurant_Reviews.tsv`
- **Link**: [Restaurant Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/vigneshwarsofficial/reviews)
- **Format**: Tab-separated values (TSV)
- **Description**: Contains 1,000+ restaurant reviews used for training and analysis
- **Note**: The dataset is included in the repository and loaded directly into the app

---

## Live Application

The application is publicly accessible via Streamlit Cloud:  
**[Launch RestoByte](https://restobyteappgit-hgk2hyraubtsjususq7vfv.streamlit.app)**

---

## Getting Started

To run the application locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/codewithkusuma/RestoByte_app.git

# Navigate to the project directory
cd RestoByte_app

# Install the required packages
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
