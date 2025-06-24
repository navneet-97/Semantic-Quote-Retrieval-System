import streamlit as st
import json
import os
import sys
from typing import Dict, List
import plotly.express as px
import pandas as pd

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGQuoteSystem

# Configure Streamlit page
st.set_page_config(
    page_title="Semantic Quote Retrieval System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .quote-card {
        color: black;
        text-weight: bold;  
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .quote-card p strong {
        color: black; 
        font-size: 1.1rem;
    }
    .author-name {
        font-weight: bold;
        color: #1f77b4;
    }
    .similarity-score {
        background-color: #e8f4fd;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
    }
    .tags {
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system"""
    try:
        rag_system = RAGQuoteSystem(
            model_path='models/fine_tuned_model',
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        rag_system.load_model_and_data('data/processed_quotes.json')
        
        # Try to load existing index, build if not found
        try:
            rag_system.load_index('models/quote_index.faiss')
        except:
            st.info("Building search index... This may take a moment.")
            rag_system.build_index()
            rag_system.save_index('models/quote_index.faiss')
        
        return rag_system
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        return None

def display_quote_card(quote_data: Dict, index: int):
    """Display a quote in a styled card format"""
    with st.container():
        st.markdown(f"""
        <div class="quote-card">
            <h4>Quote {index + 1}</h4>
            <p><strong>"{quote_data['quote']}"</strong></p>
            <p class="author-name">— {quote_data['author']}</p>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <span class="tags">Tags: {', '.join(quote_data['tags']) if quote_data['tags'] else 'None'}</span>
                <span class="similarity-score">Similarity: {quote_data['similarity_score']:.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_json_response(response_data: Dict):
    """Display the structured JSON response"""
    st.subheader("📋 Structured JSON Response")
    
    # Create a clean JSON structure for display
    display_json = {
        "query": response_data["query"],
        "total_results": response_data["total_results"],
        "llm_response": response_data.get("llm_response", ""),
        "retrieved_quotes": []
    }
    
    for quote in response_data["retrieved_quotes"]:
        display_json["retrieved_quotes"].append({
            "quote": quote["quote"],
            "author": quote["author"],
            "tags": quote["tags"],
            "similarity_score": round(quote["similarity_score"], 3)
        })
    
    st.json(display_json)

def create_analytics_dashboard(rag_system, response_data: Dict):
    """Create analytics dashboard with visualizations"""
    st.subheader("📊 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Similarity scores distribution
        scores = [quote["similarity_score"] for quote in response_data["retrieved_quotes"]]
        authors = [quote["author"] for quote in response_data["retrieved_quotes"]]
        
        fig_scores = px.bar(
            x=authors,
            y=scores,
            title="Similarity Scores by Author",
            labels={"x": "Author", "y": "Similarity Score"}
        )
        fig_scores.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        # Tags distribution
        all_tags = []
        for quote in response_data["retrieved_quotes"]:
            all_tags.extend(quote["tags"])
        
        if all_tags:
            tag_counts = pd.Series(all_tags).value_counts().head(10)
            fig_tags = px.pie(
                values=tag_counts.values,
                names=tag_counts.index,
                title="Top Tags in Results"
            )
            st.plotly_chart(fig_tags, use_container_width=True)
        else:
            st.info("No tags available for visualization")

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Semantic Quote Retrieval System</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by RAG (Retrieval Augmented Generation)**")
    
    # Load RAG system
    rag_system = load_rag_system()
    
    if not rag_system:
        st.error("Failed to load the RAG system. Please check your setup.")
        return
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Search parameters
    top_k = st.sidebar.slider("Number of results", min_value=1, max_value=20, value=5)
    use_llm = st.sidebar.checkbox("Use LLM for response generation", value=True)
    
    # Example queries
    st.sidebar.header("💡 Example Queries")
    example_queries = [
        "quotes about love and relationships",
        "Shakespeare quotes about life",
        "motivational quotes for success",
        "quotes about courage by women authors",
        "philosophical quotes about happiness",
        "quotes about friendship and loyalty"
    ]
    
    selected_example = st.sidebar.selectbox("Select an example query:", [""] + example_queries)
    
    # Main search interface
    st.header("🔍 Search for Quotes")
    
    # Query input
    query = st.text_input(
        "Enter your query:",
        value=selected_example if selected_example else "",
        placeholder="e.g., 'quotes about hope by Oscar Wilde'"
    )
    
    # Search button
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching for relevant quotes..."):
            # Perform search
            response = rag_system.search(query, top_k=top_k, use_llm=use_llm)
            
            # Display results
            if response["retrieved_quotes"]:
                # LLM Response (if available)
                if response.get("llm_response"):
                    st.header("🤖 AI-Generated Response")
                    st.markdown(response["llm_response"])
                    st.divider()
                
                # Retrieved quotes
                st.header(f"📖 Retrieved Quotes ({response['total_results']} results)")
                
                # Display quotes in tabs for better organization
                if len(response["retrieved_quotes"]) > 3:
                    tabs = st.tabs([f"Results 1-3", f"Results 4-{len(response['retrieved_quotes'])}"])
                    
                    with tabs[0]:
                        for i, quote in enumerate(response["retrieved_quotes"][:3]):
                            display_quote_card(quote, i)
                    
                    with tabs[1]:
                        for i, quote in enumerate(response["retrieved_quotes"][3:], 3):
                            display_quote_card(quote, i)
                else:
                    for i, quote in enumerate(response["retrieved_quotes"]):
                        display_quote_card(quote, i)
                
                st.divider()
                
                # JSON Response
                display_json_response(response)
                
                st.divider()
                
                # Analytics Dashboard
                create_analytics_dashboard(rag_system, response)
                
            else:
                st.warning("No quotes found for your query. Try rephrasing or using different keywords.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: white;'>
        <p>Built with Streamlit • Powered by Sentence Transformers & OpenAI</p>
        <p>📚 Semantic Quote Retrieval System with RAG</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()