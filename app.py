from flask import Flask, render_template, jsonify
import pandas as pd
import os

# 🔹 Dynamically get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 Helper function to safely load CSVs
def load_csv(relative_path):
    """Load a CSV file safely with error handling."""
    filepath = os.path.join(BASE_DIR, relative_path)
    if os.path.exists(filepath):
        return pd.read_csv(filepath, encoding="utf-8")
    print(f"⚠️ Warning: {relative_path} not found. Running without this dataset.")
    return pd.DataFrame()  # Return an empty DataFrame to prevent errors

# 🔹 Load datasets with dynamic paths
data_df = load_csv("Dataset/drugs_and_context.csv")
geo_df = load_csv("Dataset/geospatial_insights.csv")
yearly_trends = load_csv("Dataset/opioid_mentions_yearly.csv")

# 🔹 Initialize Flask app
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/heatmap-page")
def heatmap_page():
    return render_template("heatmap.html")

@app.route("/drugs")
def get_drugs():
    if "Drug" not in data_df.columns:
        return jsonify({"error": "Column 'Drug' missing in dataset"}), 500

    drugs = data_df["Drug"].dropna().unique()
    return jsonify(sorted(drugs))

@app.route("/drug-sentiments/<drug>")
def drug_sentiments(drug):
    """Return sentiment counts for a specific drug."""
    if "Drug" not in data_df.columns or "Sentiment" not in data_df.columns:
        return jsonify({"error": "Required columns missing"}), 500

    # Filter dataset for the selected drug
    filtered_data = data_df[data_df["Drug"].str.lower() == drug.lower()]

    if filtered_data.empty:
        return jsonify({"error": f"No data found for {drug}"}), 404

    # Convert numerical sentiment score into categories
    def classify_sentiment(score):
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        return "Neutral"

    # Apply classification
    filtered_data = filtered_data.copy()  # Avoid modifying original dataframe
    filtered_data["Sentiment_Label"] = filtered_data["Sentiment"].apply(classify_sentiment)

    # Count sentiment occurrences
    sentiment_counts = filtered_data["Sentiment_Label"].value_counts().to_dict()
    return jsonify(sentiment_counts)

@app.route("/trend-data/<drug>")
def trend_data(drug):
    if yearly_trends.empty:
        return jsonify({"error": "Yearly trends dataset is empty"}), 500

    # Ensure correct column names
    trend_df = yearly_trends.copy()
    trend_df.columns = trend_df.columns.str.strip().str.lower()

    if "drugs_mentioned" not in trend_df.columns:
        return jsonify({"error": "Column 'drugs_mentioned' not found"}), 500

    # Normalize drug names for case-insensitive matching
    trend_df["drugs_mentioned"] = trend_df["drugs_mentioned"].str.lower()

    # Filter for the selected drug
    filtered_df = trend_df[trend_df["drugs_mentioned"] == drug.lower()]

    if filtered_df.empty:
        return jsonify([])

    return jsonify(filtered_df.to_dict(orient="records"))

@app.route("/heatmap-data")
def heatmap_data():
    if geo_df.empty or not {"Latitude", "Longitude", "Mentions"}.issubset(geo_df.columns):
        return jsonify({"error": "Heatmap dataset missing or incorrect format"}), 500

    data = geo_df[['Latitude', 'Longitude', 'Mentions']].dropna().to_dict(orient='records')
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
