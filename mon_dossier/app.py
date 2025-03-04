import os
from dotenv import load_dotenv

load_dotenv()

# Example usage of an environment variable
api_key = os.getenv('API_KEY')
