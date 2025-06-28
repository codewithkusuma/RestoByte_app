# RestoByte: Restaurant Review Sentiment Analysis App

RestoByte is a sentiment analysis web application built using Streamlit and TextBlob. It processes restaurant reviews and classifies them into Positive, Negative, or Neutral sentiments. The app offers meaningful insights through interactive visualizations and allows users to enter their own reviews for real-time sentiment analysis.

---

## Project Overview

This project demonstrates the application of Natural Language Processing (NLP) techniques in a real-world scenario—analyzing customer sentiment in the restaurant domain.

Key functionalities include:
- Sentiment classification using TextBlob
- Real-time analysis of user-submitted reviews
- Visualization of sentiment distribution
- Word cloud generation for positive and negative reviews
- Option to download the analyzed review dataset

---

## Technologies Used

- Python  
- Streamlit  
- TextBlob  
- NLTK  
- Pandas  
- Matplotlib  
- Seaborn  
- WordCloud

---

## Dataset (from Kaggle.com)

File Name: Restaurant_Reviews.tsv 

Dataset Link: [https://www.kaggle.com/datasets/vigneshwarsofficial/reviews]

Format: Tab-separated values (TSV)

Description: Contains over 1,000 restaurant reviews used for performing sentiment analysis

Purpose: Enables classification and visualization of sentiment trends

Note: The dataset is included in the repository and is loaded directly within the app

---

## Live Application

The app is publicly deployed using Streamlit Cloud:  
[https://restobyteappgit-hgk2hyraubtsjususq7vfv.streamlit.app]

---

## Getting Started

To run the application locally:

```bash
# Clone the repository
git clone https://github.com/codewithkusuma/RestoByte_app.git

# Navigate to the project directory
cd RestoByte_app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

