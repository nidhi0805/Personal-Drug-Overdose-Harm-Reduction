import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import yake
import swifter
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dataset_path = "../Dataset"
try:
    assert os.path.exists(dataset_path), "Dataset folder does NOT exist."
    logging.info(f"Dataset folder exists at: {os.path.abspath(dataset_path)}")
except AssertionError as error:
    logging.error(error)

# Download NLTK dependencies
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) | {
    "know", "like", "thing", "think", "people", "something", "someone",
    "going", "lot", "really", "way", "one", "many", "even", "much", "still", 
    "could", "would", "make", "take", "want", "need", "get", "time", "feel",
    "use", "said", "see", "come", "also", "say", "go", "back", "well"
}

def preprocess_text(text):
    try:
        if isinstance(text, str):
            words = word_tokenize(text.lower())
            processed_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
            return " ".join(processed_words)
        return ""
    except Exception as e:
        logging.error(f"Error processing text: {e}")
        return ""

def extract_key_phrases(text, max_words=100):
    try:
        if isinstance(text, str) and len(text.split()) > 5:
            truncated_text = " ".join(text.split()[:max_words])
            kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=5)
            keywords = kw_extractor.extract_keywords(truncated_text)
            return " ".join([kw[0] for kw in keywords])
        return ""
    except Exception as e:
        logging.error(f"Error extracting key phrases: {e}")
        return ""

# Load dataset
logging.info("Loading dataset...")
df = pd.read_csv("../Dataset/combined_healthcare_reddit_data_praw.csv")

# Convert 'created' column to datetime format
df["created"] = pd.to_datetime(df["created"], errors='coerce')
df["Date"] = df["created"].dt.date

# Preprocess text
logging.info("Preprocessing text...")
df["Cleaned_Text"] = df["selftext"].swifter.apply(preprocess_text)

# Extract key phrases
logging.info("Extracting key phrases...")
df["Key_Phrases"] = df["Cleaned_Text"].swifter.apply(extract_key_phrases)

# Filter opioid-related discussions
opioid_keywords = [
    "opioid", "fentanyl", "heroin", "oxycodone", "hydrocodone", "methadone",
    "tramadol", "suboxone", "vicodin", "percocet", "oxycontin", "dilaudid"
]
df["is_opioid_related"] = df["Cleaned_Text"].apply(lambda text: any(kw in text for kw in opioid_keywords))

# Aggregate daily mentions of opioid discussions
daily_trends = df[df["is_opioid_related"]].groupby("Date").size().reset_index(name="mentions")

# Rename columns for Prophet forecasting
daily_trends.rename(columns={"Date": "ds", "mentions": "y"}, inplace=True)

#  Save cleaned text & trend data
df.to_csv("../Dataset/cleaned_healthcare_reddit_data_praw.csv", index=False)
daily_trends.to_csv("../Dataset/opioid_mentions_by_date.csv", index=False)

logging.info("Text Preprocessing Completed! Cleaned text, key phrases, and daily trends extracted.")
