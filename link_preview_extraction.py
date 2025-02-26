from dotenv import load_dotenv
import requests
import json
import os
import time
import pandas as pd
from mistralai import Mistral
from pydantic import BaseModel


load_dotenv()

api_key = os.getenv("LINK_PREVIEW_API_KEY")
api_url = 'https://api.linkpreview.net'


mistral_api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-small-latest"
client = Mistral(api_key=mistral_api_key)

class JobRoles(BaseModel):
    roles: str

def generate_job_roles(response_df):
    def generate_roles(title, description):
        prompt = f"""Based on the following title and description, generate the top 4 job roles that can be targeted for these learning paths/interview guides:
        Title: {title}
        Description: {description}
        """
        
        chat_response = client.chat.parse(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "Generate job roles."
                },
                {
                    "role": "user", 
                    "content": prompt
                },
            ],
            response_format=JobRoles,
            max_tokens=256,
            temperature=0
        )
        
        print("Raw response content:", chat_response.choices[0].message.parsed.roles)
        
        try:
            roles = chat_response.choices[0].message.parsed.roles
            return roles
        except Exception as e:
            print("Failed to extract roles:", e)
            return ''

    # Iterate through each row in the DataFrame and generate job roles
    for index, row in response_df.iterrows():
        roles = generate_roles(row['title'], row['description'])
        response_df.at[index, 'roles'] = roles

        # Add a delay to handle the API rate limit
        time.sleep(1)  # 1 second delay to ensure no more than 1 request per second

    return response_df

def extract_data_url(data):
    responses = []
    for index, key in data.iterrows():
        target = key['url']
        response = requests.get(
            api_url,
            headers={'X-Linkpreview-Api-Key': api_key},
            params={'q': target},
        )
        response_data = response.json()
        responses.append({
            'url': response_data.get('url'),
            'title': response_data.get('title'),
            'description': response_data.get('description')
        })
    
    response_df = pd.DataFrame(responses)
    return response_df

# For Link Preview API
# data = pd.read_csv('./data/learning_path.csv')
# response_df = extract_data_url(data)
# print(response_df)
# response_df.to_csv('response_data.csv', index=False)

# For Roles in Link Preview API using Mistral
data = pd.read_csv('response_data.csv')
# response_df = extract_data_url(data)
response_df_with_roles = generate_job_roles(data)
print(response_df_with_roles)
response_df_with_roles.to_csv('response_data_with_roles.csv', index=False)