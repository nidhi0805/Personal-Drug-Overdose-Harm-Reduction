
from flask import Flask, render_template, jsonify
import pandas as pd
from collections import Counter

app = Flask(__name__)

# Load datasets
data_df = pd.read_csv("Dataset/drugs_and_context.csv")
geo_df = pd.read_csv("Dataset/geospatial_insights.csv")

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
    # Ensure 'Drug' and 'Sentiment' columns exist
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


@app.route("/heatmap-data")
def heatmap_data():
    if not geo_df.empty:
        data = geo_df[['Latitude', 'Longitude', 'Mentions']].dropna().to_dict(orient='records')
        return jsonify(data)
    else:
        return jsonify([])

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=5001)
