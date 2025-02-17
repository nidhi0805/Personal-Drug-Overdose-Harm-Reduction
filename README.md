# Overview
This project analyzes Reddit discussions related to opioid use, identifying trends, sentiment analysis, and potential drug interactions. The goal is to provide insights into public discussions around opioid harm reduction, overdose trends, and associated risks.

## Features
Sentiment Analysis: Determines whether discussions about specific drugs are Positive, Negative, or Neutral.
Trend Analysis: Tracks yearly mentions of opioids and related substances on Reddit.
Forecasting: Predicts future trends in opioid mentions.
Geospatial Insights: Maps locations where opioid discussions are most active.
Heatmap Visualization: Highlights areas with high opioid-related discussions.
Drug-Specific Insights: Analyzes drug interactions & risks based on discussions.

## Dataset Details
Your project processes the following datasets:
drugs_and_context.csv	Contains Reddit posts mentioning opioid-related substances.
opioid_mentions_by_date.csv	Aggregated daily opioid mentions from Reddit.
opioid_mentions_yearly.csv	Summarized yearly opioid discussions.
geospatial_insights.csv	Contains location data for opioid-related posts.

## Application Workflow
Extract data from Reddit: Using APIs or web scraping.
Preprocess the text:
Tokenization, Lemmatization, Stopword Removal.
Extract keywords using YAKE.
Sentiment Analysis:
Classify posts into Positive, Negative, Neutral using VADER.
Trend Analysis:
Aggregate yearly mentions for different opioids.
Forecast future trends using Prophet.
Visualization:
Google Charts for Sentiment Analysis & Trends.
Heatmap for geographic insights.

## Technologies Used
Python (Flask, Pandas, Matplotlib, Prophet)
Natural Language Processing (YAKE, NLTK, VADER)
Data Visualization (Google Charts, Heatmaps)
GitHub & Render for Deployment
Reddit API for Data Collection

<img width="1438" alt="image" src="https://github.com/user-attachments/assets/c89a8946-d109-40d8-8401-33711df9caf2" />
<img width="1438" alt="image" src="https://github.com/user-attachments/assets/6598b31e-ab36-4844-8c70-a2a14f41ed35" />

