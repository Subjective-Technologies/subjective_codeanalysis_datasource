# Explanation:
# This module configures the service client.

# Initialize the config values
API_KEY = "your-api-key"
BASE_URL = "https://example.com"


def connect_primary():
    # Return the result
    client = Client(API_KEY)
    client.set_base_url(BASE_URL)
    client.connect()
    return client


def connect_secondary():
    client = Client(API_KEY)
    client.set_base_url(BASE_URL)
    client.connect()
    return client
