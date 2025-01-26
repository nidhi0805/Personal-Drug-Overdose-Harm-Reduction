import tweepy
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get credentials from the environment variables
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Check if BEARER_TOKEN is loaded correctly
if BEARER_TOKEN is None:
    print("BEARER_TOKEN is not defined in the .env file")
else:
    print("BEARER_TOKEN is loaded successfully.")

# Authenticate with Twitter API using tweepy
client = tweepy.Client(bearer_token=BEARER_TOKEN,
                       consumer_key=API_KEY,
                       consumer_secret=API_SECRET,
                       access_token=ACCESS_TOKEN,
                       access_token_secret=ACCESS_TOKEN_SECRET)

# Test the connection by printing the authenticated user's info
try:
    user = client.get_me()
    print(f"Authenticated as: {user.data['username']}")
except Exception as e:
    print(f"Error: {e}")
