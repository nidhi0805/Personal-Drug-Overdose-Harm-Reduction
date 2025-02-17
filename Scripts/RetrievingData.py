import praw
import pandas as pd
from datetime import datetime

<<<<<<< HEAD
=======

>>>>>>> nidhi
# Initialize Reddit API client
reddit = praw.Reddit(client_id='q9B_ymm5vW_3Fb-0pm-r9w', 
                     client_secret='dF8iPqxbABZ_5QPGqTrTc2Qtc9cRyg', 
                     user_agent='myredditscript')

subreddits = [
    "healthcare", "substanceabuse", "mentalhealth", "addiction", "opiates", "drugabuse", 
    "depression", "Anxiety", "medicare", "healthpolicy", "HealthcareWorkers", "pharmacy",
    "publichealth", "MentalHealthSupport", "Drugs", "OpiateRecovery","fentanyl","healthcarepolicy","AmericanDrugEpidemic"]

# List of keywords to search for
keywords = ["overdose", "opioids", "addiction", "substance abuse", "mental health", "opioid crisis","opioids", "substance abuse", "addiction", "heroin", "fentanyl", "methamphetamine",
    "cocaine", "crack cocaine", "alcoholism", "drug addiction", "drug recovery", "withdrawal",
    "detox", "rehab", "recovery", "sobriety", "relapse", "support group", "opioid crisis",
    "overdose", "opioid epidemic", "addict", "mental health", "depression", "anxiety", "bipolar",
    "schizophrenia", "PTSD", "stress", "self-harm", "suicide prevention", "therapy", "counseling",
    "psychotherapy", "emotional support", "ADHD", "OCD", "borderline personality disorder",
    "eating disorder", "schizoaffective disorder", "panic attack", "healthcare policy", "universal healthcare",
    "health insurance", "Obamacare", "single-payer healthcare", "Medicare for All", "Medicaid", 
    "Affordable Care Act", "public health", "healthcare reform", "insurance coverage", "healthcare workers",
    "nursing", "doctor shortage", "medical access", "patient advocacy", "healthcare systems", "mental health funding",
    "pharmaceutical policies", "prescription drugs", "opioid prescriptions", "over-the-counter drugs",
    "pharmaceuticals", "drug policy", "drug regulation", "drug marketing", "drug abuse prevention",
    "prescription drug misuse", "pain management", "health education"]

# Function to collect Reddit posts from given subreddits and keywords
def get_reddit_data(keywords, subreddits, limit=100):
    all_data = []
    
    # Loop through each subreddit
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        
        # Loop through each keyword and fetch posts
        for keyword in keywords:
            print(f"Collecting posts from {subreddit_name} with keyword '{keyword}'...")
            for post in subreddit.search(keyword, sort='new', time_filter='all', limit=limit):
                data = {
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'selftext': post.selftext,
                    'url': post.url,
                    'created': datetime.utcfromtimestamp(post.created_utc),
                    'keyword': keyword,
                    'score': post.score,
                    'num_comments': post.num_comments
                }
                all_data.append(data)
    
    # Convert the collected data to a DataFrame
    return pd.DataFrame(all_data)

# Collect data
reddit_data = get_reddit_data(keywords, subreddits, limit=100)

# Save the data to a CSV file
reddit_data.to_csv("combined_healthcare_reddit_data_praw.csv", index=False)
print(f"Collected {len(reddit_data)} posts from Reddit.")