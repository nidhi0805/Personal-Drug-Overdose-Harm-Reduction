import praw
import pandas as pd
from datetime import datetime

# Initialize Reddit API client
reddit = praw.Reddit(client_id='q9B_ymm5vW_3Fb-0pm-r9w', 
                     client_secret='dF8iPqxbABZ_5QPGqTrTc2Qtc9cRyg', 
                     user_agent='myredditscript')

# Specify the subreddit(s) you want to track
subreddits = ['fentanyl', 'opiates', 'overdoses']  # Example subreddits
posts_data = []

# Fetch posts in real-time
for subreddit in subreddits:
    for submission in reddit.subreddit(subreddit).new(limit=10):  # You can adjust the 'limit' as needed
        posts_data.append({
            'Title': submission.title,
            'Upvotes': submission.score,
            'Comments': submission.num_comments,
            'URL': submission.url,
            'Text': submission.selftext,
            'Created At': datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        })

# Convert to DataFrame
df = pd.DataFrame(posts_data)

# Display the dataframe
print(df.head())
