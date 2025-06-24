import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import openai
import os
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    quote: str
    author: str
    tags: List[str]
    similarity_score: float
    id: int

class RAGQuoteSystem:
    def __init__(self, model_path: str, openai_api_key: str = None):
        self.model_path = model_path
        self.model = None
        self.index = None
        self.quotes_data = []
        self.embeddings = None
        
        if openai_api_key:
            openai.api_key = openai_api_key
        
    def load_model_and_data(self, data_path: str):
        """Load the fine-tuned model and processed data"""
        logger.info("Loading fine-tuned model...")
        self.model = SentenceTransformer(self.model_path)
        
        logger.info("Loading processed quotes data...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.quotes_data = json.load(f)
        
        logger.info(f"Loaded {len(self.quotes_data)} quotes")
        
    def build_index(self):
        """Build FAISS index for quote embeddings"""
        logger.info("Building FAISS index...")
        
        # Get all quote texts
        quote_texts = [
            f"{item['quote']} - {item['author']} (Tags: {', '.join(item.get('tags', []))})"
            for item in self.quotes_data
        ]

        # Generate embeddings
        self.embeddings = self.model.encode(quote_texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
        
    def retrieve_quotes(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant quotes for a given query"""
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.quotes_data):
                quote_data = self.quotes_data[idx]
                result = RetrievalResult(
                    quote=quote_data['quote'],
                    author=quote_data['author'],
                    tags=quote_data['tags'],
                    similarity_score=float(score),
                    id=quote_data.get('id', -1)
                )
                results.append(result)
        
        return results
    
    def generate_response(self, query: str, retrieved_quotes: List[RetrievalResult]) -> str:
        """Generate response using OpenAI API with retrieved context"""
        # Prepare context from retrieved quotes
        context = "Here are some relevant quotes:\n\n"
        for i, result in enumerate(retrieved_quotes[:3], 1):
            context += f"{i}. \"{result.quote}\" - {result.author}\n"
            if result.tags:
                context += f"   Tags: {', '.join(result.tags)}\n"
            context += "\n"
        
        # Create prompt
        prompt = f"""Based on the following quotes, please provide a comprehensive answer to the user's query.

Query: {query}

{context}

Please provide a thoughtful response that:
1. Directly addresses the user's query
2. References relevant quotes from the context
3. Provides additional insights or commentary
4. Maintains a helpful and engaging tone

Response:"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides insights about quotes and helps users find meaningful quotes based on their queries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I found some relevant quotes for you, but I'm unable to generate a detailed response at the moment."
    
    def search(self, query: str, top_k: int = 5, use_llm: bool = True) -> Dict:
        """Main search function that combines retrieval and generation"""
        # Retrieve relevant quotes
        retrieved_quotes = self.retrieve_quotes(query, top_k)
        
        # Generate response if LLM is available
        llm_response = ""
        if use_llm and hasattr(openai, 'api_key') and openai.api_key:
            llm_response = self.generate_response(query, retrieved_quotes)
        
        # Format response
        response = {
            "query": query,
            "llm_response": llm_response,
            "retrieved_quotes": [
                {
                    "quote": result.quote,
                    "author": result.author,
                    "tags": result.tags,
                    "similarity_score": result.similarity_score
                }
                for result in retrieved_quotes
            ],
            "total_results": len(retrieved_quotes)
        }
        
        return response
    
    def save_index(self, index_path: str):
        """Save FAISS index to disk"""
        faiss.write_index(self.index, index_path)
        logger.info(f"Index saved to {index_path}")
        
    def load_index(self, index_path: str):
        """Load FAISS index from disk"""
        self.index = faiss.read_index(index_path)
        logger.info(f"Index loaded from {index_path}")

def main():
    # Initialize RAG system
    rag_system = RAGQuoteSystem(
        model_path='models/fine_tuned_model',
        openai_api_key=os.getenv('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') 
    )
    
    # Load model and data
    rag_system.load_model_and_data('data/processed_quotes.json')
    
    # Build index
    rag_system.build_index()
    
    # Save index for future use
    rag_system.save_index('models/quote_index.faiss')
    
    # Test queries
    test_queries = [
        "quotes about love and relationships",
        "Shakespeare quotes about life",
        "motivational quotes for success",
        "quotes about courage by women authors"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        result = rag_system.search(query, top_k=3)
        
        if result['llm_response']:
            print("LLM Response:")
            print(result['llm_response'])
            print()
        
        print("Retrieved Quotes:")
        for i, quote in enumerate(result['retrieved_quotes'], 1):
            print(f"{i}. \"{quote['quote']}\" - {quote['author']}")
            print(f"   Similarity: {quote['similarity_score']:.3f}")
            if quote['tags']:
                print(f"   Tags: {', '.join(quote['tags'])}")
            print()

if __name__ == "__main__":
    main()