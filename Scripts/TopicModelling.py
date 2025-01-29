from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# Initialize SentenceTransformer model for topic modeling and embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
topic_model = BERTopic(embedding_model=embedding_model)

# Load cleaned Reddit data
df = pd.read_csv("/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/Individual XN-Repo/Drug-Overdose-Harm-Reduction/Dataset/cleaned_healthcare_reddit_data_praw.csv")

# Ensure 'Cleaned_Text' column contains valid strings (not NaN or floats)
df['Cleaned_Text'] = df['Cleaned_Text'].fillna('').astype(str)

# Fit the topic model on the cleaned text
topics, probs = topic_model.fit_transform(df['Cleaned_Text'])

# Get topics and associated words
topics_dict = topic_model.get_topics()

# Function to get associated topics for a dynamic keyword using semantic search (BERT or Word2Vec)
def get_associated_topics(keyword, topics_dict, word2vec_model=None, top_n=5):
    associated_topics = []
    keyword_lower = keyword.lower()  # Convert to lowercase to handle case-insensitivity
    
    # Generate word embeddings for the keyword using BERT or Word2Vec
    if word2vec_model:
        try:
            keyword_embedding = word2vec_model.wv[keyword_lower]
        except KeyError:
            print(f"Keyword '{keyword}' not found in Word2Vec model.")
            return []
    else:
        # Generate BERT embeddings for the keyword
        keyword_embedding = embedding_model.encode([keyword_lower])[0]

    # Iterate through each topic and check if the keyword is part of any topic's words
    for topic_num, topic_words in topics_dict.items():
        topic_word_embeddings = []
        for word, weight in topic_words:
            word_lower = word.lower()
            
            if word_lower == keyword_lower:  # Skip the keyword itself to avoid self-matching
                continue
            
            # Generate embeddings for words in the topic using BERT or Word2Vec
            if word2vec_model:
                try:
                    word_embedding = word2vec_model.wv[word_lower]
                except KeyError:
                    word_embedding = np.zeros(100)  # Use zero vector if the word is not in Word2Vec model
            else:
                word_embedding = embedding_model.encode([word_lower])[0]

            # Compute cosine similarity between keyword and each word in the topic
            cosine_sim = cosine_similarity([keyword_embedding], [word_embedding])[0][0]
            if cosine_sim > 0.5:  # Threshold to only keep relevant words (can be adjusted)
                topic_word_embeddings.append((word, weight, cosine_sim))
        
        # Sort by cosine similarity and weight
        if topic_word_embeddings:
            topic_word_embeddings.sort(key=lambda x: (x[1], x[2]), reverse=True)  # Sort by weight and similarity
            top_associated_words = topic_word_embeddings[:top_n]  # Get top N words for the topic
            
            # Create a description of the topic based on the keyword
            topic_description = f"Topic {topic_num}: " + ", ".join([f"{word} (Weight: {weight:.4f}, CosSim: {sim:.4f})" for word, weight, sim in top_associated_words])
            associated_topics.append(topic_description)
    
    # Return results (or message if no associations were found)
    if not associated_topics:
        return [f"No associated topics found for keyword '{keyword}'."]
    
    return associated_topics

# Example of how you'd call the function in your app (just replace 'keyword' with actual input):
# keyword = "overdose"  # You can dynamically pass this from your app's frontend or logic
# associated_topics = get_associated_topics(keyword, topics_dict, word2vec_model=None)

# To use the function, you'd call it like:
# associated_topics = get_associated_topics('overdose', topics_dict)

# For testing, print the associated topics for a specific keyword
# For example, replace 'overdose' with any keyword you'd like to search
associated_topics = get_associated_topics('overdose', topics_dict)
print(f"Associated topics for 'overdose':")
for topic in associated_topics:
    print(topic)

topics_list = []
for topic_num, topic_words in topics_dict.items():
    for word, weight in topic_words:
        cos_sim = None  # Initialize CosSim as None
        
        # Compute Cosine Similarity with the keyword "overdose" (or another reference word)
        word_embedding = embedding_model.encode([word.lower()])[0]
        keyword_embedding = embedding_model.encode(["overdose"])[0]  # Use any reference keyword
        
        cos_sim = cosine_similarity([keyword_embedding], [word_embedding])[0][0]  # Compute similarity
        
        topics_list.append({
            "Topic": topic_num,
            "Word": word,
            "Weight": weight,
            "CosSim": cos_sim  # Store computed CosSim
        })

topics_df = pd.DataFrame(topics_list)
topics_df.to_csv("topics.csv", index=False)

# Optionally print the first few topics for inspection
print(topics_df.head())
