import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

# Function to preprocess the text data
def preprocess_text(text):
    # Check if the text is a string and not a NaN or float
    if isinstance(text, str):
        # Tokenize text
        words = word_tokenize(text.lower())
        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        # Remove words that are stopwords, not alphabetic, or contain numbers
        filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
        return ' '.join(filtered_words)
    else:
        # If not a string (e.g., NaN or float), return an empty string
        return ''

# Example usage to preprocess the Reddit posts
df = pd.read_csv("combined_healthcare_reddit_data_praw.csv")
df['Cleaned_Text'] = df['selftext'].apply(preprocess_text)

# Save the cleaned data to a new CSV file
df.to_csv("cleaned_healthcare_reddit_data_praw.csv", index=False)
