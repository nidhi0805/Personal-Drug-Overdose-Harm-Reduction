import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Load cleaned Reddit data
df = pd.read_csv("cleaned_healthcare_reddit_data_praw.csv")

# Ensure 'Cleaned_Text' column contains only valid strings
df['Cleaned_Text'] = df['Cleaned_Text'].fillna('').astype(str)

# Function to get sentiment scores
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']

# Apply sentiment analysis to each post
df['Sentiment_Score'] = df['Cleaned_Text'].apply(get_sentiment)

# Categorize sentiment
df['Sentiment_Label'] = df['Sentiment_Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

# Save the sentiment results to a new CSV
df.to_csv("sentiment_healthcare_reddit_data.csv", index=False)

print("Sentiment analysis completed and saved to 'sentiment_healthcare_reddit_data.csv'.")
