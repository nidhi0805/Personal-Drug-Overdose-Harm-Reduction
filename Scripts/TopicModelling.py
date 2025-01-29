from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize SentenceTransformer model and BERTopic model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
topic_model = BERTopic(embedding_model=embedding_model)

# Load cleaned Reddit data
df = pd.read_csv("cleaned_healthcare_reddit_data_praw.csv")

# Ensure 'Cleaned_Text' column contains only valid strings (not NaN or floats)
df['Cleaned_Text'] = df['Cleaned_Text'].fillna('').astype(str)

# Fit the topic model on the cleaned text
topics, probs = topic_model.fit_transform(df['Cleaned_Text'])

# Get topics and associated words
topics_dict = topic_model.get_topics()

# Print and save the topics
for topic_num, topic_words in topics_dict.items():
    print(f"Topic {topic_num}: {[word for word, _ in topic_words]}")

# Save the topics to a CSV file
topics_df = pd.DataFrame(topics_dict)
topics_df.to_csv("topics.csv", index=False)
