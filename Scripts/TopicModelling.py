from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize SentenceTransformer model and BERTopic model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
topic_model = BERTopic(embedding_model=embedding_model)

# Load cleaned Reddit data
df = pd.read_csv("/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/Individual XN-Repo/Drug-Overdose-Harm-Reduction/Dataset/cleaned_healthcare_reddit_data_praw.csv")

# Ensure 'Cleaned_Text' column contains only valid strings (not NaN or floats)
df['Cleaned_Text'] = df['Cleaned_Text'].fillna('').astype(str)

# Fit the topic model on the cleaned text
topics, probs = topic_model.fit_transform(df['Cleaned_Text'])

# Get topics and associated words
topics_dict = topic_model.get_topics()

# Function to get associated topics for a dynamic keyword
def get_associated_topics(keyword, topics_dict, top_n=5):
    associated_topics = []
    keyword_lower = keyword.lower()  # Convert to lowercase to handle case-insensitivity
    
    # Iterate through each topic and check if the keyword is part of any topic's words
    for topic_num, topic_words in topics_dict.items():
        associated_words = []
        
        # Check if the keyword or any related word exists in the topic
        for word, weight in topic_words:
            if keyword_lower in word.lower():  # Checking for matches (case insensitive)
                associated_words.append((word, weight))
        
        # If associated words are found for this topic, sort them by weight and limit the result
        if associated_words:
            associated_words.sort(key=lambda x: x[1], reverse=True)  # Sort by weight (high to low)
            top_associated_words = associated_words[:top_n]  # Get top N words for the topic
            
            # Create a description of the topic based on the keyword
            topic_description = f"Topic {topic_num}: " + ", ".join([f"{word} ({weight:.4f})" for word, weight in top_associated_words])
            associated_topics.append(topic_description)
    
    # Return results (or message if no associations were found)
    if not associated_topics:
        return [f"No associated topics found for keyword '{keyword}'."]
    
    return associated_topics

# Example of taking input for keyword dynamically
keyword = input("Enter the keyword to find associated topics: ")

# Get associated topics for the entered keyword
associated_topics = get_associated_topics(keyword, topics_dict)

# Display the associated topics for the entered keyword
print(f"Associated topics for '{keyword}':")
for topic in associated_topics:
    print(topic)

# Optional: Save topics to a CSV file for reference
topics_list = []
for topic_num, topic_words in topics_dict.items():
    for word, weight in topic_words:
        topics_list.append({"Topic": topic_num, "Word": word, "Weight": weight})

topics_df = pd.DataFrame(topics_list)
topics_df.to_csv("topics.csv", index=False)

# Optionally print the first few topics for inspection
print(topics_df.head())
