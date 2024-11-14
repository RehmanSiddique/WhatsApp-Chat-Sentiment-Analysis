import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
 

 # Function to generate a sentiment configuration matrix
def generate_sentiment_matrix(df, pos_threshold, neg_threshold, aggregation_level):
    # Initialize Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Calculate sentiment scores for each message
    df['Positive'] = df['Message'].apply(lambda x: sia.polarity_scores(str(x))['pos'])
    df['Negative'] = df['Message'].apply(lambda x: sia.polarity_scores(str(x))['neg'])
    df['Neutral'] = df['Message'].apply(lambda x: sia.polarity_scores(str(x))['neu'])
    
    # Classify sentiment based on the thresholds
    df['Sentiment'] = df.apply(lambda row: 'Positive' if row['Positive'] > pos_threshold else (
        'Negative' if row['Negative'] > neg_threshold else 'Neutral'), axis=1)
    
    # Aggregate data based on the selected level (User or Date)
    if aggregation_level == 'User':
        sentiment_matrix = df.groupby('User')[['Positive', 'Negative', 'Neutral']].mean()
    elif aggregation_level == 'Date':
        sentiment_matrix = df.groupby(df['Date'].dt.date)[['Positive', 'Negative', 'Neutral']].mean()
    else:
        st.error("Invalid aggregation level. Please select 'User' or 'Date'.")
        return None
    
    return sentiment_matrix

# Plotting the heatmap
def plot_sentiment_heatmap(matrix, title="Sentiment Configuration Matrix Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Sentiment Categories")
    ax.set_ylabel("Aggregation Level")
    st.pyplot(fig)

def get_sentiment(df):
    df['Date']=pd.to_datetime(df['Date'])
    data=df.dropna()
    sentiments=SentimentIntensityAnalyzer()
    data["positive"]=[sentiments.polarity_scores(i)["pos"] for i in df["Message"]]
    data["negative"]=[sentiments.polarity_scores(i)["neg"] for i in df["Message"]]
    data["neutral"]=[sentiments.polarity_scores(i)["neu"] for i in df["Message"]]
    # data["positive"]=abs( data["positive"]) 
    x=sum(data["positive"])
    y=sum(data["negative"])
    z=sum(data["neutral"])
    sizes = [x,y,z]
    labels = ['Positive chat','Negative chat', 'Neutral chat']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes ,labels=labels, autopct='%1.1f%%')
    st.pyplot(fig1)
    # Calculate the total for percentage calculation
    # Streamlit Title
    st.title("Sentiment Distribution Matrix (Percentage)")
    total = x + y + z

    # Create a sentiment percentage matrix
    matrix_data = np.array([[x / total * 100, y / total * 100, z / total * 100]])

    # Create a DataFrame for better visualization
    df_matrix = pd.DataFrame(matrix_data, columns=["Positive", "Negative", "Neutral"])
    fig, ax = plt.subplots(figsize=(8, 6))
    # Define a custom color map for the matrix (green for positive, red for negative, gray for neutral)
    colors = ["#8BC34A", "#8B0000", "#9E9E9E"]  # Green, Red, Gray for Positive, Negative, Neutral
    sns.set(font_scale=1.5)

    # Create a heatmap with annotations, adding % sign to each value
  
    sns.heatmap(df_matrix, annot=df_matrix.applymap(lambda v: f'{v:.1f}%'), fmt='', cmap="coolwarm", cbar=False)
    ax.set_title("Sentiment Distribution Matrix (Percentage)")
    ax.set_xlabel("Sentiments")
    ax.set_ylabel("Percentage (%)")
    st.pyplot(fig)
# Function to generate sentiment scores and configuration matrix
def get_sentiment(df, pos_threshold, neg_threshold):
    # Check if 'Message' column exists, else raise an error
    if 'Message' not in df.columns:
        st.error("The uploaded file must contain a 'Message' column.")
        return

    # Drop rows with missing values in 'Message' column and reset index
    data = df.dropna(subset=['Message']).copy()

    # Initialize Sentiment Analyzer
    sentiments = SentimentIntensityAnalyzer()

    # Calculate sentiment scores and add them to the DataFrame
    data["positive"] = data["Message"].apply(lambda x: sentiments.polarity_scores(str(x))["pos"])
    data["negative"] = data["Message"].apply(lambda x: sentiments.polarity_scores(str(x))["neg"])
    data["neutral"] = data["Message"].apply(lambda x: sentiments.polarity_scores(str(x))["neu"])

    # Classify messages based on thresholds
    data['Sentiment'] = data.apply(
        lambda row: 'Positive' if row['positive'] > pos_threshold else (
            'Negative' if row['negative'] > neg_threshold else 'Neutral'
        ),
        axis=1
    )

    # Count sentiment occurrences
    sentiment_counts = data['Sentiment'].value_counts()

    # Plot Sentiment Distribution Pie Chart
    fig1, ax1 = plt.subplots()
    ax1.pie(
        sentiment_counts, labels=sentiment_counts.index,
        autopct='%1.1f%%', startangle=90, colors=['#8BC34A', '#8B0000', '#9E9E9E']
    )
    ax1.axis('equal')
    st.pyplot(fig1)

    # Display Sentiment Configuration Matrix
    configuration_matrix(sentiment_counts)


# Function to create a configuration matrix visualization
def configuration_matrix(sentiment_counts):
    # Data for the matrix
    matrix_data = np.array([[
        sentiment_counts.get('Positive', 0),
        sentiment_counts.get('Negative', 0),
        sentiment_counts.get('Neutral', 0)
    ]])

    # Create DataFrame for heatmap
    df_matrix = pd.DataFrame(matrix_data, columns=["Positive", "Negative", "Neutral"])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_matrix, annot=True, fmt='d', cmap="coolwarm", cbar=False)
    ax.set_title("Sentiment Configuration Matrix (Count)")
    st.pyplot(fig)