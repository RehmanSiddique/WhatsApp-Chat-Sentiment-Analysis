# WhatsApp-Sentiment-Analysis-
This app provides insights and statistical analysis of WhatsApp chat exports, enabling you to explore different aspects of the chat data, such as user activity, sentiment, emojis used, word frequency, and more. It leverages Natural Language Processing (NLP) techniques to analyze the sentiment of messages and visualize key metrics, like message counts, media shared, and links.
# Features
###  1. Chat Statistics
<img src="https://github.com/user-attachments/assets/87fc2a16-74a4-4e17-976c-ce93d8488ed5" alt="Image"  />
### 2. User Activity
<img src="https://github.com/user-attachments/assets/b4ad604d-489b-485e-9433-c0ff82f8c024" alt="Image"  />
      
###   3. Emoji Analysis
<img src="https://github.com/user-attachments/assets/6ed105f6-3750-491f-832b-d2f78e2e4983" alt="Image"  />
###  4. Most Common Words
<img src="https://github.com/user-attachments/assets/6c9c2e29-578a-4513-ad8b-75287116ed8c" alt="Image"  />
### 5. Monthly & Daily Timeline
<img src="https://github.com/user-attachments/assets/8091b35e-b419-4a5b-bf27-57e466dba01b" alt="Image"  />
### 6. WordCloud
<img src="https://github.com/user-attachments/assets/f5269224-62d4-4cd5-adc9-01459110443f" alt="Image"  />
###  7. Sentiment Analysis
<img src="https://github.com/user-attachments/assets/f5ee47b2-7a9e-4ace-a01c-2293623ca659" alt="Image"  />
### 8. Weekly Activity Heatmap
       <img src="https://github.com/user-attachments/assets/4c4b9781-ad79-462b-8d39-98a74a660ea1" alt="Image"  />
 
     
# Installation
To use this application, follow the instructions below.
## 1. Install Required Packages
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

The required libraries are:

### streamlit : For building the web app interface.
### pandas :For data manipulation and analysis.
### nltk :For Natural Language Processing (Sentiment Analysis).
### seaborn, matplotlib :  For data visualization.
### urlextract :For extracting URLs from messages.
### emoji :For extracting and counting emojis in messages.
### wordcloud –For generating the word cloud.
  
# Usage
## 1. Run the Streamlit app
  Once the required libraries are installed, you can run the Streamlit app by using the following command:
  streamlit run app.py
## 2. Upload WhatsApp Chat Data
  After running the app, you will be prompted to upload a WhatsApp chat export file (in .txt format). You can obtain this file by:

 1. Open WhatsApp.
 2. Go to the chat you want to analyze.
 3. Tap on the chat name > Export Chat > Without Media.
 4. Download the .txt file.
## 3. Analyze the Data
Once the chat file is uploaded, you will be able to explore the following:
    i. User Selection
   ii. Statistics:  
   iii. Activity: 
   iv. WordCloud:  
   v. Sentiment:  
# Analysis Types
## 1. Chat Statistics
  Displays the total number of messages, total words, media shared, links, and deleted messages.
## 2. User Activity
  Displays the number of messages sent by each user, along with the percentage of the total messages.
## 3. Emoji Usage
  Shows the most frequently used emojis and their count in the chat.

##  4. Common Words
  Lists the most frequent words used in the chat (excluding stopwords like "the", "and", etc.).

## 5. Monthly Timeline
  Shows a plot of messages sent per month, helping to visualize chat activity over time.

## 6. Daily Timeline
  Displays a line chart showing messages sent per day.

## 7. Weekly Activity Heatmap
  A heatmap displaying user activity throughout the week by hour and day.

## 8. Sentiment Analysis
  Categorizes messages as positive, negative, or neutral based on the content of the message, and displays the results as a pie chart and a sentiment matrix.
 


