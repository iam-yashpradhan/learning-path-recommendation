import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
host = os.getenv("PINECONE_HOST")

# Initialize Pinecone and SentenceTransformer model
pc = Pinecone(api_key=api_key)
index = pc.Index(host=host)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set page configuration
st.set_page_config(
    page_title="Career Resource Recommender",
    page_icon="📚",
    layout="wide"
)

# Define categories
CATEGORIES = ["learning-path", "blog", "interview guide"]

# Common roles for suggestions
COMMON_ROLES = [
    "Data Scientist", 
    "Data Analyst", 
    "Machine Learning", 
    "Data Engineer",
    "Business Intelligence Analyst",
    "Software Engineer",
    "Product Analyst",
    "Quantitative Analyst",
    "Business Analyst"
]

# Function to get recommendations for a role
def get_recommendations(role, top_k=20):
    # Encode the query
    xq = model.encode(role).tolist()
    
    # Query Pinecone
    xc = index.query(vector=xq, top_k=top_k, include_metadata=True)
    
    # If no matches found, return empty results
    if not xc['matches']:
        return []
    
    # Transform documents for reranking
    transformed_documents = [
        {
            'id': match['id'],
            'reranking_field': '; '.join([f"{key}: {value}" for key, value in match['metadata'].items()])
        }
        for match in xc['matches']
    ]
    
    try:
        # Perform reranking
        reranked_results = pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=role,
            documents=transformed_documents,
            rank_fields=["reranking_field"],
            top_n=top_k,
            return_documents=True,
        )
        
        # Check if reranking returned documents
        if hasattr(reranked_results, 'documents') and reranked_results.documents:
            # Extract original metadata from reranked results
            results_with_metadata = []
            for doc in reranked_results.documents:
                doc_id = doc.id
                # Find the original metadata from xc
                for match in xc['matches']:
                    if match['id'] == doc_id:
                        results_with_metadata.append({
                            'id': doc_id,
                            'metadata': match['metadata']
                        })
                        break
            return results_with_metadata
        else:
            # If reranking didn't work, fall back to original results
            return [{'id': match['id'], 'metadata': match['metadata']} for match in xc['matches']]
    except Exception as e:
        st.error(f"Error during reranking: {e}")
        # If reranking fails, fall back to original results
        return [{'id': match['id'], 'metadata': match['metadata']} for match in xc['matches']]

# Function to categorize recommendations
def categorize_recommendations(results):
    categorized = {category: [] for category in CATEGORIES}
    
    for result in results:
        metadata = result['metadata']
        category = metadata.get('category', '')
        
        # Add to appropriate category if it exists
        if category and category in categorized:
            categorized[category].append({
                'title': metadata.get('title', 'No title'),
                'description': metadata.get('description', 'No description'),
                'url': metadata.get('url', '#'),
                'roles': metadata.get('roles', [])
            })
    
    return categorized

# Function to create a downloadable CSV from resources
def create_download_csv(resources_by_category):
    # Create a list to hold all resources
    all_resources = []
    
    # Add category to each resource and collect them
    for category, resources in resources_by_category.items():
        for resource in resources:
            # Create a new dict with all the data we want in the CSV
            resource_data = {
                'Category': category,
                'Title': resource['title'],
                'Description': resource['description'],
                'URL': resource['url']
            }
            # Add roles as a comma-separated string if available
            if isinstance(resource.get('roles'), list):
                resource_data['Relevant Roles'] = ', '.join(resource['roles'])
            else:
                resource_data['Relevant Roles'] = ''
                
            all_resources.append(resource_data)
            
    # Create DataFrame
    df = pd.DataFrame(all_resources)
    
    return df

# Title and description
st.title("Career Resource Recommender")
st.markdown("Get personalized learning resources based on your target role")

# Sidebar for role selection
with st.sidebar:
    st.header("Select Target Role")
    
    # Option to choose from common roles or enter custom role
    
    
    
    role = st.selectbox("Target Roles", COMMON_ROLES)
    
    
    # Number of results to show
    results_count = st.number_input("Number of results per category", min_value=1, max_value=20, value=5)
    
    # Button to get recommendations
    st.button("Get Recommendations", type="primary", use_container_width=True, key="sidebar_button")

# Alternative placement in main area
if not st.session_state.get("sidebar_button", False):
    col1, col2 = st.columns([2, 1])
    with col1:
        # Only show this if not using the sidebar button
        if "role" not in locals():
            role = st.text_input("Or enter your target role directly:", "Data Scientist")
            get_recs_button = st.button("Get Recommendations", key="main_button")
        else:
            get_recs_button = False
    
    

# Combine button states
get_recommendations_clicked = st.session_state.get("sidebar_button", False) or ("get_recs_button" in locals() and get_recs_button)

# Get recommendations when button is clicked
if get_recommendations_clicked and role:
    with st.spinner('Finding the best resources for you...'):
        # Get recommendations
        results = get_recommendations(role)
        
        if not results:
            st.warning(f"No results found for the role '{role}'. Try a different role or check your Pinecone setup.")
        else:
            # Categorize results
            categorized_results = categorize_recommendations(results)
            
            # Create download data
            if any(categorized_results.values()):
                resources_df = create_download_csv(categorized_results)
                csv = resources_df.to_csv(index=False)
                
                # Add download button at the top
                st.download_button(
                    label="Download all resources as CSV",
                    data=csv,
                    file_name=f"{role.replace(' ', '_')}_resources.csv",
                    mime="text/csv",
                )
            
            # Display recommendations by category in tabs
            tabs = st.tabs([f"{category.title()} " for category in CATEGORIES])
            
            for i, category in enumerate(CATEGORIES):
                with tabs[i]:
                    # If no recommendations in this category
                    if not categorized_results.get(category, []):
                        st.info(f"No {category} resources found for this role.")
                        continue
                    
                    resources = categorized_results[category]
                    
                    # Display counter of total resources
                    st.caption(f"Showing {min(len(resources), results_count)} of {len(resources)} resources")
                    
                    # Display top N recommendations for this category
                    for resource in resources[:results_count]:
                        with st.container():
                            st.markdown(f"#### [{resource['title']}]({resource['url']})")
                            
                            
                            # Display roles this resource is relevant for
                            if isinstance(resource.get('roles'), list) and resource['roles']:
                                st.caption(f"**Relevant for:** {', '.join(resource['roles'])}")
                        
                        st.divider()
                    
                    # Show more button
                    if len(resources) > results_count:
                        with st.expander(f"Show {len(resources) - results_count} more"):
                            for resource in resources[results_count:]:
                                st.markdown(f"- [{resource['title']}]({resource['url']})")

