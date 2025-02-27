import re
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import seaborn as sns
import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
import urlextract
import emoji
from wordcloud import WordCloud
import numpy as np
       
# Function to generate DataFrame from the uploaded WhatsApp chat file
def generateDataFrame(file):
    data = file.read().decode("utf-8")  # Read and decode the file
    data = data.replace('\u202f', ' ')  # Replace special spaces with normal spaces
    data = data.replace('\n', ' ')  # Remove newlines
    
    # Regex pattern to split messages based on date and time
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?(?:AM\s|PM\s|am\s|pm\s)?-\s'
    msgs = re.split(pattern, data)[1:]  # Split chat into messages
    date_times = re.findall(pattern, data)  # Extract date and time
    
    # Separate date and time into lists
    date = []
    time = []
    for dt in date_times:
        date.append(re.search('\d{1,2}/\d{1,2}/\d{2,4}', dt).group())
        time.append(re.search('\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?', dt).group())

    # Separate users and messages
    users = []
    message = []
    for m in msgs:
        entry = re.split('([\w\W]+?):\s', m)
        if len(entry) < 3:  # If system notification (e.g., joined/left group)
            users.append("Notifications")
            message.append(entry[0])
        else:  # Otherwise, extract user and message
            users.append(entry[1])
            message.append(entry[2])
    
    # Create and return a DataFrame
    df = pd.DataFrame(list(zip(date, time, users, message)), columns=["Date", "Time(U)", "User", "Message"])
    return df

# Function to get a list of users from the DataFrame
def getUsers(df):
    users = df['User'].unique().tolist()  # Get unique users
    users.sort()  # Sort users alphabetically
    users.remove('Notifications')  # Remove system notifications
    users.insert(0, 'Everyone')  # Add 'Everyone' option at the beginning
    return users

# Function to extract statistics such as media, deleted messages, and links
def getStats(df):
    # Count and drop media messages
    media = df[df['Message'] == "<Media omitted> "]
    media_cnt = media.shape[0]
    df.drop(media.index, inplace=True)
    
    # Count and drop deleted messages
    deleted_msgs = df[df['Message'] == "This message was deleted "]
    deleted_msgs_cnt = deleted_msgs.shape[0]
    df.drop(deleted_msgs.index, inplace=True)
    
    # Remove system notifications
    temp = df[df['User'] == 'Notifications']
    df.drop(temp.index, inplace=True)
    
    # Extract links using URLExtract
    extractor = urlextract.URLExtract()
    links = []
    for msg in df['Message']:
        x = extractor.find_urls(msg)
        if x:
            links.extend(x)
    links_cnt = len(links)
    
    # Count total words and messages
    word_list = []
    for msg in df['Message']:
        word_list.extend(msg.split())
    word_count = len(word_list)
    msg_count = df.shape[0]
    
    return df, media_cnt, deleted_msgs_cnt, links_cnt, word_count, msg_count

# Function to extract emojis from messages
def getEmoji(df):
    emojis = []
    for message in df['Message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])  # Check each character for emojis
    return pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))  # Count emojis

# Function to preprocess data by extracting year, month, day, and time
def PreProcess(df, dayf):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=dayf)  # Convert to datetime
    df['Time'] = pd.to_datetime(df['Time(U)']).dt.time  # Extract time
    
    # Extract year, month, day, and other details
    df['year'] = df['Date'].apply(lambda x: int(str(x)[:4]))
    df['month'] = df['Date'].apply(lambda x: int(str(x)[5:7]))
    df['date'] = df['Date'].apply(lambda x: int(str(x)[8:10]))
    df['day'] = df['Date'].apply(lambda x: x.day_name())
    df['hour'] = df['Time'].apply(lambda x: int(str(x)[:2]))
    df['month_name'] = df['Date'].apply(lambda x: x.month_name())
    return df

# Function to create monthly message timeline
def getMonthlyTimeline(df):
    df.columns = df.columns.str.strip()  # Remove any whitespace in column names
    df = df.reset_index()
    
    # Group messages by year and month
    timeline = df.groupby(['year', 'month']).count()['Message'].reset_index()
    
    # Create a time label for each year-month pair
    time = []
    for i in range(timeline.shape[0]):
        time.append(str(timeline['month'][i]) + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

# Function to extract most common words from messages
def MostCommonWords(df):
    # Load stop words
    f = open('stop_hinglish.txt')
    stop_words = f.read()
    f.close()
    
    words = []
    for message in df['Message']:
        for word in message.lower().split():
            if word not in stop_words:  # Exclude stop words
                words.append(word)
    return pd.DataFrame(Counter(words).most_common(20))  # Return top 20 words

# Function to display daily message timeline
def dailytimeline(df):
    df['taarek'] = df['Date']
    daily_timeline = df.groupby('taarek').count()['Message'].reset_index()
    
    # Plot daily timeline
    fig, ax = plt.subplots()
    ax.plot(daily_timeline['taarek'], daily_timeline['Message'])
    ax.set_ylabel("Messages Sent")
    st.title('Daily Timeline')
    st.pyplot(fig)

# Function to analyze weekly activity
def WeekAct(df):
    x = df['day'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(x.index, x.values)
    ax.set_xlabel("Days")
    ax.set_ylabel("Message Sent")
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

# Function to analyze monthly activity
def MonthAct(df):
    x = df['month_name'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(x.index, x.values)
    ax.set_xlabel("Months")
    ax.set_ylabel("Message Sent")
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

# Function to generate an activity heatmap
def activity_heatmap(df):
    period = []
    for hour in df[['day', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    user_heatmap = df.pivot_table(index='day', columns='period', values='Message', aggfunc='count').fillna(0)
    return user_heatmap

# Function to create a word cloud from messages
def create_wordcloud(df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    f.close()
    # Helper function to remove stop words from messages
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    df['Message'] = df['Message'].apply(remove_stop_words)
    df_wc = wc.generate(df['Message'].str.cat(sep=" "))
    return df_wc  
 