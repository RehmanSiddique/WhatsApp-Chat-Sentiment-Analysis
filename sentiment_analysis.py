import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Ensure the VADER lexicon is downloaded
import nltk
nltk.download('vader_lexicon')
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer  



# Function to generate a sentiment configuration matrix
def generate_sentiment_matrix(df, pos_threshold, neg_threshold, aggregation_level):
     
    # Initialize Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()

    # Calculate sentiment scores for each message and store as new columns
    df['Positive'] = df['Message'].apply(lambda x: sia.polarity_scores(str(x))['pos'])
    df['Negative'] = df['Message'].apply(lambda x: sia.polarity_scores(str(x))['neg'])
    df['Neutral'] = df['Message'].apply(lambda x: sia.polarity_scores(str(x))['neu'])

    # Classify messages based on thresholds
    df['Sentiment'] = df.apply(
        lambda row: 'Positive' if row['Positive'] > pos_threshold else
        ('Negative' if row['Negative'] > neg_threshold else 'Neutral'), axis=1
    )

    # Aggregate sentiment scores at the specified level
    if aggregation_level == 'User':
        sentiment_matrix = df.groupby('User')[['Positive', 'Negative', 'Neutral']].mean()
    else:
        st.error("Invalid aggregation level. Please select 'User' or 'Date'.")
        return None

    return sentiment_matrix


# Function to plot sentiment heatmap
def plot_sentiment_heatmap(matrix, title="Sentiment Configuration Matrix Heatmap"):
    """
    Generate a heatmap from the sentiment matrix.

    Args:
        matrix (DataFrame): Sentiment configuration matrix.
        title (str): Title for the heatmap plot.
    """
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot heatmap with annotations
    sns.heatmap(matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5, ax=ax)

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel("Sentiment Categories")
    ax.set_ylabel("Aggregation Level")

    # Display the plot in Streamlit
    st.pyplot(fig)


# Function to calculate and display sentiment analysis results
def get_sentiment(df):
    """
    Perform sentiment analysis on the chat data and display the sentiment distribution.

    Args:
        df (DataFrame): Input DataFrame with 'Message' and 'Date' columns.
    """
    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Drop rows with missing messages
    data = df.dropna()

    # Initialize Sentiment Analyzer
    sentiments = SentimentIntensityAnalyzer()

    # Calculate sentiment scores for each message
    data["positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["Message"]]
    data["negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["Message"]]
    data["neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["Message"]]

    # Calculate total sentiment scores for positive, negative, and neutral messages
    x = sum(data["positive"])
    y = sum(data["negative"])
    z = sum(data["neutral"])
    sizes = [x, y, z]
    labels = ['Positive chat', 'Negative chat', 'Neutral chat']

    # Plot sentiment distribution pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
    st.pyplot(fig1)

    # Title for sentiment distribution matrix
    st.title("Sentiment Distribution Matrix (Percentage)")

    # Calculate percentages for each sentiment
    total = x + y + z
    matrix_data = np.array([[x / total * 100, y / total * 100, z / total * 100]])

    # Create DataFrame for heatmap
    df_matrix = pd.DataFrame(matrix_data, columns=["Positive", "Negative", "Neutral"])

    # Plot sentiment percentage heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_matrix, annot=df_matrix.applymap(lambda v: f'{v:.1f}%'), fmt='', cmap="coolwarm", cbar=False)
    ax.set_title("Sentiment Distribution Matrix (Percentage)")
    ax.set_xlabel("Sentiments")
    ax.set_ylabel("Percentage (%)")
    st.pyplot(fig)


# Function to classify and visualize sentiment configuration
def get_sentiment(df, pos_threshold, neg_threshold):
    """
    Analyze sentiment for messages, plot a pie chart, and generate a heatmap.

    Args:
        df (DataFrame): Input DataFrame with a 'Message' column.
        pos_threshold (float): Threshold for positive sentiment.
        neg_threshold (float): Threshold for negative sentiment.
    """
    # Check if the DataFrame contains 'Message' column
    if 'Message' not in df.columns:
        st.error("The uploaded file must contain a 'Message' column.")
        return

    # Drop missing values in 'Message' column
    data = df.dropna(subset=['Message']).copy()

    # Initialize Sentiment Analyzer
    sentiments = SentimentIntensityAnalyzer()

    # Calculate sentiment scores for each message
    data["positive"] = data["Message"].apply(lambda x: sentiments.polarity_scores(str(x))["pos"])
    data["negative"] = data["Message"].apply(lambda x: sentiments.polarity_scores(str(x))["neg"])
    data["neutral"] = data["Message"].apply(lambda x: sentiments.polarity_scores(str(x))["neu"])

    # Classify sentiment based on thresholds
    data['Sentiment'] = data.apply(
        lambda row: 'Positive' if row['positive'] > pos_threshold else
        ('Negative' if row['negative'] > neg_threshold else 'Neutral'), axis=1
    )

    # Count occurrences of each sentiment
    sentiment_counts = data['Sentiment'].value_counts()

    # Plot sentiment distribution pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(
        sentiment_counts, labels=sentiment_counts.index,
        autopct='%1.1f%%', startangle=90, colors=['#8BC34A', '#8B0000', '#9E9E9E']
    )
    ax1.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    st.pyplot(fig1)

    # Display sentiment configuration matrix
    configuration_matrix(sentiment_counts)


# Function to plot a configuration matrix
def configuration_matrix(sentiment_counts):
    """
    Plot a heatmap for sentiment distribution counts.

    Args:
        sentiment_counts (Series): Counts of each sentiment type.
    """
    # Prepare data for heatmap
    matrix_data = np.array([[
        sentiment_counts.get('Positive', 0),
        sentiment_counts.get('Negative', 0),
        sentiment_counts.get('Neutral', 0)
    ]])

    # Convert data to DataFrame for plotting
    df_matrix = pd.DataFrame(matrix_data, columns=["Positive", "Negative", "Neutral"])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_matrix, annot=True, fmt='d', cmap="coolwarm", cbar=False)
    ax.set_title("Sentiment Configuration Matrix (Count)")
    st.pyplot(fig)
 