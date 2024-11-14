import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import functions
import sentiment_analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer


st.title('WhatsApp Chat Analyzer')
file = st.file_uploader("Choose a file")

if file:
    df = functions.generateDataFrame(file)
    try:
        users = functions.getUsers(df)
        users_s = st.sidebar.selectbox("Select User to View Analysis", users)
        selected_user=""

        if st.sidebar.button("Show Analysis"):
            selected_user = users_s

            st.title("Showing Reults for : " + selected_user)
            df = functions.PreProcess(df,True)
            if selected_user != "Everyone":
                df = df[df['User'] == selected_user]
            df, media_cnt, deleted_msgs_cnt, links_cnt, word_count, msg_count = functions.getStats(df)
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

            # User Activity Count
            if selected_user == 'Everyone':
                x = df['User'].value_counts().head()
                name = x.index
                count = x.values
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

            # Emoji
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
            # Common Word
            commonWord = functions.MostCommonWords(df)
            fig, ax = plt.subplots()
            ax.bar(commonWord[0], commonWord[1])
            ax.set_xlabel("Words")
            ax.set_ylabel("Frequency")
            plt.xticks(rotation='vertical')
            st.title('Most Frequent Words Used In Chat')
            st.pyplot(fig)

            # Monthly Timeline
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

            st.title('Most Busy Days')
            functions.WeekAct(df)
            st.title('Most Busy Months')
            functions.MonthAct(df)

            # WordCloud
            st.title("Wordcloud")
            df_wc = functions.create_wordcloud(df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            st.title("Weekly Activity Map")
            user_heatmap = functions.activity_heatmap(df)
            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)
             
            # Sentiment Analysis
     
    
            # Sidebar options for configuring thresholds
            st.sidebar.header("Sentiment Thresholds")
            pos_threshold = st.sidebar.slider("Positive Threshold", 0.0, 1.0, 0.3)
            neg_threshold = st.sidebar.slider("Negative Threshold", 0.0, 1.0, 0.3)
    
             # Select aggregation level (e.g., User or Date)
            aggregation_level = st.sidebar.selectbox("Aggregate by", ["User", "Date"])
    
            # Generate and display the sentiment matrix
            sentiment_matrix = sentiment_analysis.generate_sentiment_matrix(df, pos_threshold, neg_threshold, aggregation_level)
            if sentiment_matrix is not None:
                sentiment_analysis.plot_sentiment_heatmap(sentiment_matrix) 
            st.sidebar.header("Sentiment Analysis Configuration") 
            st.subheader("Sentiment Analysis with Configurable Thresholds")
            sentiment_analysis.get_sentiment(df, pos_threshold, neg_threshold) 
             

    except Exception as e:
        st.subheader("Unable to Process Your Request")
