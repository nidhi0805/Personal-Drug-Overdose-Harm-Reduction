# import pandas as pd
# import logging
# from nltk.sentiment import SentimentIntensityAnalyzer
# import nltk

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Ensure required NLTK downloads
# nltk.download('vader_lexicon')

# # Initialize the VADER sentiment analyzer
# sia = SentimentIntensityAnalyzer()

# def classify_sentiment(text):
#     scores = sia.polarity_scores(text)
#     return 'positive' if scores['compound'] > 0 else 'negative'

# # Load the cleaned dataset
# df = pd.read_csv("../Dataset/cleaned_healthcare_reddit_data_praw.csv")

# # Define a comprehensive list of drugs related to overdoses and illicit drug issues
# drug_list = [
#     'Fentanyl', 'Heroin', 'Morphine', 'Oxycodone', 'Hydrocodone', 'Codeine',
#     'Methadone', 'Tramadol', 'Carfentanil', 'Vicodin', 'Percocet', 'OxyContin',
#     'Dilaudid', 'Buprenorphine', 'Suboxone', 'Methamphetamine', 'Cocaine',
#     'Xanax', 'Alprazolam', 'Benzodiazepine', 'Ketamine', 'MDMA', 'Ecstasy'
# ]

# def find_drugs_in_text(text):
#     found_drugs = [drug for drug in drug_list if drug.lower() in text.lower()]
#     return found_drugs

# # Apply sentiment classification to cleaned_text
# df['Sentiment'] = df['cleaned_text'].apply(classify_sentiment)

# # Extract drugs mentioned in cleaned_text
# df['Drugs_Mentioned'] = df['cleaned_text'].apply(find_drugs_in_text)

# # Expand the list of drugs into separate rows for easy analysis
# rows = []
# _ = df.apply(lambda row: [rows.append([row['cleaned_text'], drug, row['Sentiment']]) for drug in row['Drugs_Mentioned']], axis=1)
# drugs_df = pd.DataFrame(rows, columns=['Text', 'Drug', 'Context'])

# # Save the processed data
# output_path = "../Dataset/drugs_and_context.csv"
# if drugs_df.empty:
#     logging.warning("No drugs found in any cleaned_text. Resulting CSV will be empty.")
# else:
#     drugs_df.to_csv(output_path, index=False)
#     logging.info(f"Processed data saved to {output_path}")

import pandas as pd
import logging
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('vader_lexicon')

# Initialize the VADER Sentiment Analyzer **before** using it
sia = SentimentIntensityAnalyzer()

# Initialize Hugging Face transformer model for sentiment analysis
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load the cleaned dataset
df = pd.read_csv("../Dataset/cleaned_healthcare_reddit_data_praw.csv")

# Define a comprehensive list of drugs related to overdoses and illicit drug issues
drug_list = [
    'Fentanyl', 'Heroin', 'Morphine', 'Oxycodone', 'Hydrocodone', 'Codeine',
    'Methadone', 'Tramadol', 'Carfentanil', 'Vicodin', 'Percocet', 'OxyContin',
    'Dilaudid', 'Buprenorphine', 'Suboxone', 'Methamphetamine', 'Cocaine',
    'Xanax', 'Alprazolam', 'Benzodiazepine', 'Ketamine', 'MDMA', 'Ecstasy'
]

def find_drugs_in_text(text):
    if not isinstance(text, str):  # Check if text is not a string
        return []  # Return an empty list if it's not a string (e.g., NaN)
    
    found_drugs = [drug for drug in drug_list if drug.lower() in text.lower()]
    return found_drugs


def get_sentiment(text):
    if isinstance(text, str):  # Check if the value is a string
        text = text.strip()
        if text == '':
            return 0  # Neutral sentiment for empty strings
        scores = sia.polarity_scores(text)
        return scores['compound']
    return 0  # Default to neutral sentiment if not a string

# Apply sentiment classification to cleaned text
df['Sentiment'] = df['Cleaned_Text'].apply(get_sentiment)

# Extract drugs mentioned in cleaned_text
df['Drugs_Mentioned'] = df['Cleaned_Text'].apply(find_drugs_in_text)

# Expand the list of drugs into separate rows for easy analysis
rows = []
_ = df.apply(lambda row: [rows.append([row['Cleaned_Text'], drug, row['Sentiment']]) for drug in row['Drugs_Mentioned']], axis=1)
drugs_df = pd.DataFrame(rows, columns=['Text', 'Drug', 'Sentiment'])

# Save the processed data
output_path = "../Dataset/drugs_and_context.csv"
if drugs_df.empty:
    logging.warning("No drugs found in any cleaned_text. Resulting CSV will be empty.")
else:
    drugs_df.to_csv(output_path, index=False)
    logging.info(f"Processed data saved to {output_path}")
