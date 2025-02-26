import os
from mistralai import Mistral
from pydantic import BaseModel
import pandas as pd
import ast
import time
import random
from dotenv import load_dotenv

load_dotenv()

class Blog(BaseModel):
    url: str
    blog_title: str
    category: str
    roles: str

# Your API key and model details
api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-small-latest"

# Load the CSV file
data = pd.read_csv("./data/iq_links.csv")
client = Mistral(api_key=api_key)

# Function to generate data fields using the Mistral API with exponential backoff
def generate_data_fields(row, max_retries=5):
    prompt = f"""Based on the URL, extract the title and store it in blog_title schema. If the URL has
     /p/ in it, then add blog in the category schema, 
     if it has /interview-guide/, then add interview guide in the category schema. 
    Also, based on the title, analyze for which job roles might it help in data science and the job role in roles schema: {row.to_dict()}"""

    initial_wait_time = 1  # Initial wait time in seconds for backoff

    for retry in range(max_retries):
        try:
            # Call the Mistral API
            chat_response = client.chat.parse(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate some data fields."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                response_format=Blog,
                max_tokens=256,
                temperature=0
            )
            return chat_response  # Return response if successful

        except client.models.sdkerror.SDKError as e:
            # Check if it's a rate limit error (429)
            if e.status_code == 429:
                wait_time = initial_wait_time * (2 ** retry) + random.uniform(0, 1)  # Exponential backoff with jitter
                print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # Raise an exception for non-rate-limit errors
                raise Exception(f"API call failed with error: {e}")

    # Raise an exception if max retries are exceeded
    raise Exception("Max retries exceeded. Could not process the request.")

# Iterate through each row in the DataFrame and generate data fields
for index, row in data.iterrows():
    try:
        generated_data = generate_data_fields(row)
        # Assuming the generated data is a dictionary, update the DataFrame with the new fields
        for key, value in generated_data.model_dump().items():
            if key not in data.columns:
                data[key] = None  # Add new column if it doesn't exist
            data.at[index, key] = value

        time.sleep(1)  # Add a delay between iterations to reduce request frequency

    except Exception as e:
        print(f"Error processing row {index}: {e}")

# Print and save the updated DataFrame to a CSV file
print(data)
data.to_csv('test.csv', index=False)
