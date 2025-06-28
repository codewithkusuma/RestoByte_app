import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
nltk.download('punkt', quiet=True)  # Required for TextBlob tokenization

# ----- Page Setup -----
st.set_page_config(page_title="RestoByte Sentiment Dashboard", layout="wide", page_icon="🍽️")
st.markdown("""
    <style>
        .adaptive-header {
            text-align: center;
            font-size: 2em;
            font-weight: bold;
        }
        @media (prefers-color-scheme: dark) {
            .adaptive-header {
                color: white;
            }
        }
        @media (prefers-color-scheme: light) {
            .adaptive-header {
                color: black;
            }
        }
    </style>
    <h1 class='adaptive-header'>🍽️ RestoByte: A Sentiment Analysis on Restaurant Reviews</h1>
""", unsafe_allow_html=True)


# ----- Custom Button Styling -----
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #105a8b;
    }
    </style>
""", unsafe_allow_html=True)

# ----- Sentiment Functions -----
def textblob_sentiment(text):
    return TextBlob(text).sentiment.polarity

def classify_sentiment(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to validate if review is restaurant-related
def is_restaurant_related(text):
    keywords = [
        'food', 'restaurant', 'taste', 'service', 'meal', 'waiter',
        'ambience', 'menu', 'cuisine', 'dish', 'dining', 'chef'
    ]
    return any(word in text.lower() for word in keywords)

# ----- Load Dataset -----
@st.cache_data
def load_data():
    df = pd.read_csv("Restaurant_Reviews.tsv", sep='\t')
    df['Polarity'] = df['Review'].apply(textblob_sentiment)
    df['Sentiment'] = df['Polarity'].apply(classify_sentiment)
    return df

# ----- WordCloud Generator -----
@st.cache_data
def generate_wordcloud(text):
    return WordCloud(width=600, height=400, background_color='white').generate(text)

# Load the data
df = load_data()

# ----- Dataset Preview -----
with st.expander(" Click to Preview Dataset"):
    st.dataframe(df.head(), use_container_width=True)

# ----- Summary Stats -----
st.markdown("### Sentiment Summary")
sentiment_counts = df['Sentiment'].value_counts()
st.markdown(f"**Total Reviews:** {len(df)}")
st.markdown(f"✅ Positive: {sentiment_counts.get('Positive', 0)}")
st.markdown(f"❌ Negative: {sentiment_counts.get('Negative', 0)}")
st.markdown(f"😐 Neutral: {sentiment_counts.get('Neutral', 0)}")

# ----- Row 1: Sentiment Visualizations -----
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Sentiment Distribution ")
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=['lightgreen', 'salmon', 'skyblue'])
    st.pyplot(fig1)

with col2:
    st.subheader("📊 Sentiment Count ")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Sentiment', data=df, hue='Sentiment', palette='Set2', ax=ax2, legend=False)
    st.pyplot(fig2)

# ----- Row 2: User Input Section -----
st.markdown("---")
col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("📝 Try It Yourself")
    user_input = st.text_area("Write a restaurant-related review here:")

    if st.button(" Classify Sentiment"):
        if user_input:
            if is_restaurant_related(user_input):
                with st.spinner("Analyzing your review..."):
                    polarity = textblob_sentiment(user_input)
                    sentiment = classify_sentiment(polarity)
                    emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}
                    st.success(f"**Sentiment:** {sentiment} {emoji[sentiment]}")
                    st.info(f"**Polarity Score:** {round(polarity, 2)}")

                    # Add temporary review to dataset (in-session only)
                    new_row = pd.DataFrame({'Review': [user_input], 'Polarity': [polarity], 'Sentiment': [sentiment]})
                    df = pd.concat([df, new_row], ignore_index=True)
            else:
                st.warning("⚠️ Please enter a restaurant-related review (e.g., food, service, ambience, dish, etc.).")
        else:
            st.warning("⚠️ Please enter a review before clicking the button.")

with col4:
    st.markdown("""
        ###  About the Analyzer:
        - This tool is designed specifically for restaurant review analysis
        - Avoid entering unrelated reviews like movies or tech products
        - Example: *“The food was amazing and the service was quick!”*
    """)

# ----- Download Button -----
st.markdown("---")
st.subheader("📥 Download Your Analyzed Data")
st.download_button("Download CSV", df.to_csv(index=False), file_name="sentiment_results.csv", mime="text/csv")

# ----- Row 3: Word Clouds -----
col5, col6 = st.columns([1, 1])

with col5:
    st.subheader("☁️ Positive Review Word Cloud")
    positive_text = " ".join(df[df['Sentiment'] == 'Positive']['Review'])
    wordcloud_pos = generate_wordcloud(positive_text)
    st.image(wordcloud_pos.to_array())

with col6:
    st.subheader("☁️ Negative Review Word Cloud")
    negative_text = " ".join(df[df['Sentiment'] == 'Negative']['Review'])
    wordcloud_neg = generate_wordcloud(negative_text)
    st.image(wordcloud_neg.to_array())
