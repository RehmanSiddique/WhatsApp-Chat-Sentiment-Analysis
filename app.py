import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import functions  # Custom module for core functions
import sentiment_analysis  # Custom module for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Title of the Streamlit Web App
st.title('WhatsApp Chat Sentiment Analyzer')

# Upload a file through Streamlit
file = st.file_uploader("Choose a file")

# Check if the file is uploaded
if file:
    # Generate DataFrame using a helper function from 'functions'
    df = functions.generateDataFrame(file)
    try:
        # Get the list of users from the DataFrame
        users = functions.getUsers(df)

        # User selection for analysis via sidebar
        users_s = st.sidebar.selectbox("Select User to View Analysis", users)
        selected_user = ""

        # Button to show analysis results
        if st.sidebar.button("Show Analysis"):
            selected_user = users_s

            # Display the selected user's name
            st.title("Showing Results for : " + selected_user)

            # Preprocess the data and clean up date/time
            df = functions.PreProcess(df, True)

            # Filter data for the selected user, if not 'Everyone'
            if selected_user != "Everyone":
                df = df[df['User'] == selected_user]

            # Extract chat statistics
            df, media_cnt, deleted_msgs_cnt, links_cnt, word_count, msg_count = functions.getStats(df)

            # Display Chat Statistics using columns
            st.title("Chat Statistics")
            stats_c = ["Total Messages", "Total Words", "Media Shared", "Links Shared", "Messages Deleted"]
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.subheader(stats_c[0])
                st.title(msg_count)
            with c2:
                st.subheader(stats_c[1])
                st.title(word_count)
            with c3:
                st.subheader(stats_c[2])
                st.title(media_cnt)
            with c4:
                st.subheader(stats_c[3])
                st.title(links_cnt)
            with c5:
                st.subheader(stats_c[4])
                st.title(deleted_msgs_cnt)

            # User Activity Analysis
            if selected_user == 'Everyone':
                x = df['User'].value_counts().head()
                name = x.index
                count = x.values

                # Display messaging frequency with table and bar chart
                st.title("Messaging Frequency")
                st.subheader('Messaging Percentage Count of Users')
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(round((df['User'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
                        columns={'User': 'name', 'count': 'percent'}))
                with col2:
                    fig, ax = plt.subplots()
                    ax.bar(name, count)
                    ax.set_xlabel("Users")
                    ax.set_ylabel("Message Sent")
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

            # Emoji Analysis
            emojiDF = functions.getEmoji(df)
            st.title("Emoji Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(emojiDF)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emojiDF[1].head(), labels=emojiDF[0].head(), autopct="%0.2f", shadow=True)
                plt.legend()
                st.pyplot(fig)

            # Most Common Words Analysis
            commonWord = functions.MostCommonWords(df)
            fig, ax = plt.subplots()
            ax.bar(commonWord[0], commonWord[1])
            ax.set_xlabel("Words")
            ax.set_ylabel("Frequency")
            plt.xticks(rotation='vertical')
            st.title('Most Frequent Words Used In Chat')
            st.pyplot(fig)

            # Monthly Timeline of Messages
            timeline = functions.getMonthlyTimeline(df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['Message'])
            ax.set_xlabel("Month")
            ax.set_ylabel("Messages Sent")
            plt.xticks(rotation='vertical')
            st.title('Monthly Timeline')
            st.pyplot(fig)

            # Daily Timeline
            functions.dailytimeline(df)

            # Weekly Activity Analysis
            st.title('Most Busy Days')
            functions.WeekAct(df)
            st.title('Most Busy Months')
            functions.MonthAct(df)

            # WordCloud of the most frequent words
            st.title("Wordcloud")
            df_wc = functions.create_wordcloud(df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            # Weekly Activity Map (Heatmap)
            st.title("Weekly Activity Map")
            user_heatmap = functions.activity_heatmap(df)
            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

            # Sentiment Analysis Configuration
            st.sidebar.header("Sentiment Thresholds")
            pos_threshold = st.sidebar.slider("Positive Threshold", 0.0, 1.0, 0.3)
            neg_threshold = st.sidebar.slider("Negative Threshold", 0.0, 1.0, 0.3)

            # Aggregation Level for Sentiment Analysis
            aggregation_level = st.sidebar.selectbox("Aggregate by", ["User"])

            # Generate and display sentiment matrix heatmap
            sentiment_matrix = sentiment_analysis.generate_sentiment_matrix(df, pos_threshold, neg_threshold, aggregation_level)
            if sentiment_matrix is not None:
                sentiment_analysis.plot_sentiment_heatmap(sentiment_matrix)

            # Display sentiment analysis pie chart and heatmap
            st.sidebar.header("Sentiment Analysis Configuration")
            st.subheader("Sentiment Analysis ")
            sentiment_analysis.get_sentiment(df, pos_threshold, neg_threshold)
             

    # Catch any errors and display a friendly message
    except Exception as e:
        st.subheader("Unable to Process Your Request")
