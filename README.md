# Career Resource Recommender

A Streamlit application that recommends career resources based on a user's target role. The app leverages Pinecone's vector database and reranking capabilities to find the most relevant learning resources across different categories.

## Features

- Role-based resource recommendations
- Resources categorized by type (learning paths, blogs, interview guides)
- Reranking algorithm for improved relevance
- Clean, user-friendly interface

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Pinecone API credentials:
   ```
   PINECONE_API_KEY=your_api_key
   PINECONE_HOST=your_index_host
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

## How It Works

1. User inputs their target role (e.g., "Data Scientist", "Machine Learning Engineer")
2. The app encodes the role using a SentenceTransformer model
3. It queries the Pinecone vector database for relevant documents
4. Results are reranked using Pinecone's reranking API
5. Resources are categorized and displayed by type

## Technologies Used

- Streamlit: Frontend web application
- Pinecone: Vector database and reranking
- SentenceTransformer: Text embedding
- Python: Core programming language
