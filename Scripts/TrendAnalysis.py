import pandas as pd
import matplotlib.pyplot as plt

# Load dataset with dates & texts
df_reddit = pd.read_csv("../Dataset/cleaned_healthcare_reddit_data_praw.csv")

# Load the opioid mentions dataset
df_mentions = pd.read_csv("../Dataset/opioid_mentions_by_date.csv")

# Convert dates to datetime format
df_reddit['created'] = pd.to_datetime(df_reddit['created'], errors='coerce')
df_mentions['ds'] = pd.to_datetime(df_mentions['ds'], errors='coerce')

# Extract year from created dates
df_reddit['year'] = df_reddit['created'].dt.year

# Define a list of known drugs (same as used in `drugs_and_context.csv`)
drug_list = [
    'Fentanyl', 'Heroin', 'Morphine', 'Oxycodone', 'Hydrocodone', 'Codeine',
    'Methadone', 'Tramadol', 'Carfentanil', 'Vicodin', 'Percocet', 'OxyContin',
    'Dilaudid', 'Buprenorphine', 'Suboxone', 'Methamphetamine', 'Cocaine',
    'Xanax', 'Alprazolam', 'Benzodiazepine', 'Ketamine', 'MDMA', 'Ecstasy'
]

# Function to extract drugs mentioned in each post
def find_drugs(text):
    if isinstance(text, str):
        return [drug for drug in drug_list if drug.lower() in text.lower()]
    return []

df_reddit['Drugs_Mentioned'] = df_reddit['Cleaned_Text'].apply(find_drugs)

# Explode to get one drug per row
df_drugs = df_reddit.explode('Drugs_Mentioned')

# Group by year and drug to count mentions
yearly_trends = df_drugs.groupby(['year', 'Drugs_Mentioned']).size().reset_index(name='mentions')

# Save updated dataset
yearly_trends.to_csv("../Dataset/opioid_mentions_yearly.csv", index=False)