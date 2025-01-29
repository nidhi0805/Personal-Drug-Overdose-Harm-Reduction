import os
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/Individual XN-Repo/Drug-Overdose-Harm-Reduction/Templates')


# Load the datasets (you can modify the file paths as needed)
combined_data = pd.read_csv('/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/Individual XN-Repo/Drug-Overdose-Harm-Reduction/Dataset/combined_healthcare_reddit_data_praw.csv',encoding='utf-8')
cleaned_data = pd.read_csv('/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/Individual XN-Repo/Drug-Overdose-Harm-Reduction/Dataset/cleaned_healthcare_reddit_data_praw.csv',encoding='utf-8')
sentiment_data = pd.read_csv('/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/Individual XN-Repo/Drug-Overdose-Harm-Reduction/Dataset/sentiment_healthcare_reddit_data.csv',encoding='utf-8')
topics_data = pd.read_csv('/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/Individual XN-Repo/Drug-Overdose-Harm-Reduction/Dataset/topics.csv',encoding='utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    keyword = request.form['keyword']
    
    # 1. Preprocess the keyword (if needed, such as lowercasing or cleaning)
    cleaned_keyword = keyword.lower()  # Simple example
    
    # 2. Filter relevant data based on the keyword
    relevant_rows = cleaned_data[cleaned_data['Cleaned_Text'].str.contains(cleaned_keyword, case=False, na=False)]  # Modify 'text_column' to the actual column name
    
    # 3. Extract the topics and sentiment related to the keyword
    relevant_sentiment_data = sentiment_data[sentiment_data['Cleaned_Text'].str.contains(cleaned_keyword, case=False, na=False)]  # Modify 'text_column'
    sentiment_analysis = relevant_sentiment_data['Sentiment_Label'].value_counts().to_dict()  # Modify 'text_column'
    sentiment_analysis = {key: int(value) for key, value in sentiment_analysis.items()}  # Convert to Python native int

    topics = []
    for _, row in topics_data.iterrows():
        if keyword in str(row['Word']).lower():
            topics.append(row['Word'])  # Just append topic name (no weights in response)

    associated_people = int(relevant_rows['num_comments'].sum())
    
    # Example response format
    response = {
        'keyword': keyword,
        'topics': topics,
        'sentiment': sentiment_analysis,
        'associated_people': associated_people  # How many rows are relevant to the keyword
    }

    return jsonify(response)

def filter_topics(topics_df, keyword):
    """
    Filter topics data and return relevant topics based on the search keyword.
    """
    filtered_topics = {}

    for _, row in topics_df.iterrows():
        # Ensure the value is a string before calling .lower() and convert weights to native Python types
        word = str(row['Word'])  # Convert the word to a string
        weight = row['Weight'] if isinstance(row['Weight'], (int, float)) else float(row['Weight'])

        if keyword in word.lower():  # Perform the check
            filtered_topics[word] = weight  # Store the word and its weight

    return filtered_topics

if __name__ == '__main__':
    app.run(debug=True)
