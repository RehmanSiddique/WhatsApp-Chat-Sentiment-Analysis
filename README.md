# WhatsApp-Sentiment-Analysis-
This app provides insights and statistical analysis of WhatsApp chat exports, enabling you to explore different aspects of the chat data, such as user activity, sentiment, emojis used, word frequency, and more. It leverages Natural Language Processing (NLP) techniques to analyze the sentiment of messages and visualize key metrics, like message counts, media shared, and links.
<video src="https://github.com/user-attachments/assets/0d43f59d-1ebf-47d8-95fe-354fd3b3e30e" controls width="300"></video>

# Features
##  1. Chat Statistics
<img src="https://github.com/user-attachments/assets/7a4c7c18-de1c-44b1-8a33-4fac6b62960e" alt="Image"  />

##  2. User Activity
<img src="https://github.com/user-attachments/assets/3de46fe9-4d65-4b4d-a838-1bbcc28ea8b7" alt="Image"  />

##  3. Emoji Analysis
<img src="https://github.com/user-attachments/assets/e3fc0e4c-11ba-47e2-a17b-a73619ad7314" alt="Image"  />

##  4. Most Common Words
<img src="https://github.com/user-attachments/assets/d6e606a0-fb45-485e-b4bc-9aeff8e78e64" alt="Image"  />

##  5. Monthly & Daily Timeline
<img src="https://github.com/user-attachments/assets/6b168161-ae14-4ce8-b6a3-c79225381e5a" alt="Image" width="400"  />
<img src="https://github.com/user-attachments/assets/06d3060f-a8df-4ff5-86f3-c91d3cc73c52" alt="Image" width="400"  />

##  6. WordCloud
<img src="https://github.com/user-attachments/assets/17c53bd4-f98e-4e03-9e9b-827775721628" alt="Image"  />

##  7. Sentiment Analysis
<img src="https://github.com/user-attachments/assets/a0b827e3-cb5e-4df7-99cb-c2d68e7ed8d8" alt="Image" width="400"  />
<img src="https://github.com/user-attachments/assets/d663bf81-a3d6-4951-a7d3-c4873a92d907" alt="Image" width="400"  />

##  8. Weekly Activity Heatmap
<img src="https://github.com/user-attachments/assets/c76db211-eb8a-4a39-a8b7-514f63c52b80" alt="Image"  />
 
     
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
 


