
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


from flask import Flask, render_template, jsonify
import pandas as pd
from collections import Counter

app = Flask(__name__)

# Load datasets
data_df = pd.read_csv("Dataset/drugs_and_context.csv")
geo_df = pd.read_csv("Dataset/geospatial_insights.csv")
yearly_trends = pd.read_csv("Dataset/opioid_mentions_yearly.csv")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/heatmap-page")
def heatmap_page():
    return render_template("heatmap.html")

@app.route("/drugs")
def get_drugs():
    drugs = data_df['Drug'].dropna().unique()
    return jsonify(sorted(drugs))



@app.route("/drug-sentiments/<drug>")
def drug_sentiments(drug):
    """Return sentiment counts for a specific drug."""
    # Ensure Drug and Sentiment columns exist
    if "Drug" not in data_df.columns or "Sentiment" not in data_df.columns:
        return jsonify({"error": "Required columns missing"}), 500

    # Debugging logs
    print(f"Fetching sentiment for drug: {drug}")

    # Filter dataset for the selected drug
    filtered_data = data_df[data_df["Drug"].str.lower() == drug.lower()]

    if filtered_data.empty:
        print(f"No data found for drug: {drug}")
        return jsonify({"error": "No data found for this drug"}), 404

    # Convert numerical sentiment score into categories
    def classify_sentiment(score):
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    # Apply sentiment classification
    filtered_data["Sentiment_Label"] = filtered_data["Sentiment"].apply(classify_sentiment)

    # Count sentiment occurrences
    sentiment_counts = filtered_data["Sentiment_Label"].value_counts().to_dict()

    print(f"Sentiment counts for {drug}: {sentiment_counts}")
    return jsonify(sentiment_counts)

@app.route("/trend-data/<drug>")
def trend_data(drug):
    drug = drug.lower()

    # 🔹 Print column names for debugging
    print("Columns in yearly_trends:", yearly_trends.columns)

    # 🔹 Ensure correct column names
    yearly_trends.columns = yearly_trends.columns.str.strip().str.lower()

    # 🔹 Print columns again after conversion
    print("Updated Columns:", yearly_trends.columns)

    # 🔹 Check if "drugs_mentioned" exists (lowercased version)
    if "drugs_mentioned" not in yearly_trends.columns:
        return jsonify({"error": "Column 'drugs_mentioned' not found"}), 500

    # 🔹 Convert column to lowercase for case-insensitive comparison
    yearly_trends["drugs_mentioned"] = yearly_trends["drugs_mentioned"].str.lower()

    # 🔹 Filter for the selected drug
    filtered_df = yearly_trends[yearly_trends["drugs_mentioned"] == drug]

    # 🔹 If no data found, return an empty list
    if filtered_df.empty:
        print(f"No trend data found for {drug}")
        return jsonify([])

    # 🔹 Convert to JSON
    trend_json = filtered_df.to_dict(orient="records")
    print(f"✅ Returning trend data for {drug}: {trend_json}")
    return jsonify(trend_json)

@app.route("/heatmap-data")
def heatmap_data():
    if not geo_df.empty:
        data = geo_df[['Latitude', 'Longitude', 'Mentions']].dropna().to_dict(orient='records')
        return jsonify(data)
    else:
        return jsonify([])

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=5001)

