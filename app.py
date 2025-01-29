import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/Individual XN-Repo/Drug-Overdose-Harm-Reduction/Templates')

# Load the datasets
combined_data = pd.read_csv('/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/Individual XN-Repo/Drug-Overdose-Harm-Reduction/Dataset/combined_healthcare_reddit_data_praw.csv', encoding='utf-8')
cleaned_data = pd.read_csv('/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/Individual XN-Repo/Drug-Overdose-Harm-Reduction/Dataset/cleaned_healthcare_reddit_data_praw.csv', encoding='utf-8')
sentiment_data = pd.read_csv('/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/Individual XN-Repo/Drug-Overdose-Harm-Reduction/Dataset/sentiment_healthcare_reddit_data.csv', encoding='utf-8')
topics_data = pd.read_csv('/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/Individual XN-Repo/Drug-Overdose-Harm-Reduction/Dataset/topics.csv', encoding='utf-8')

# Initialize the SentenceTransformer model for BERT-based embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get associated topics for a keyword using BERT embeddings
def get_associated_topics(keyword, topics_df, top_n=5):
    keyword_lower = keyword.lower()  # Convert to lowercase to handle case-insensitivity
    keyword_embedding = embedding_model.encode([keyword_lower])[0]

    associated_topics = []

    for _, row in topics_df.iterrows():
        word = str(row['Word']).lower()  # Ensure it's a string and lowercase
        word_embedding = embedding_model.encode([word])[0]
        
        # Compute cosine similarity between keyword and topic word
        cosine_sim = cosine_similarity([keyword_embedding], [word_embedding])[0][0]
        
        # If cosine similarity is greater than 0.5, add to associated topics list
        if cosine_sim > 0.5:  # Adjust threshold as needed
            # Include both CosSim and Weight in the description
            topic_description = f"{word} (CosSim: {cosine_sim:.4f}, Weight: {row['Weight']:.4f})"
            associated_topics.append(topic_description)

    return associated_topics[:top_n]  # Limit to top N associated topics

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    keyword = request.form['keyword']
    
    # Preprocess the keyword (lowercase)
    cleaned_keyword = keyword.lower()

    # 1. Filter relevant data based on the keyword
    relevant_rows = cleaned_data[cleaned_data['Cleaned_Text'].str.contains(cleaned_keyword, case=False, na=False)]
    
    # 2. Extract sentiment data related to the keyword
    relevant_sentiment_data = sentiment_data[sentiment_data['Cleaned_Text'].str.contains(cleaned_keyword, case=False, na=False)]
    sentiment_analysis = relevant_sentiment_data['Sentiment_Label'].value_counts().to_dict()
    sentiment_analysis = {key: int(value) for key, value in sentiment_analysis.items()}  # Convert to Python native int

    # 3. Extract topics related to the keyword from the topics_data using BERT similarity
    associated_topics = get_associated_topics(cleaned_keyword, topics_data)

    # 4. Calculate associated people (based on the number of relevant rows)
    associated_people = int(relevant_rows['num_comments'].sum())

    # Prepare the response
    response = {
        'keyword': keyword,
        'topics': associated_topics,
        'sentiment': sentiment_analysis,
        'associated_people': associated_people
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
