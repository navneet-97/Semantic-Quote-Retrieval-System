import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import logging
from typing import List, Tuple
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuoteEmbeddingTrainer:
    def __init__(self, base_model_name: str = "all-MiniLM-L6-v2"):
        self.base_model_name = base_model_name
        self.model = None
        self.training_data = []
        
    def load_processed_data(self, filepath: str):
        """Load processed quotes data"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} quotes for training")
        
    def create_training_examples(self) -> List[InputExample]:
        """Create training examples for sentence transformer"""
        examples = []
        
        # Create positive pairs (query-quote pairs)
        for quote_data in self.data:
            quote = quote_data['quote']
            author = quote_data['author']
            tags = quote_data['tags']
            
            # Create various query formats
            queries = [
                f"quotes by {author}",
                f"quotes about {' '.join(tags[:2])}" if tags else f"quotes by {author}",
                f"{author} quotes",
                f"inspirational quotes" if any(tag in ['inspiration', 'motivational', 'life'] 
                                             for tag in tags) else f"quotes by {author}"
            ]
            
            for query in queries:
                examples.append(InputExample(texts=[query, quote], label=1.0))
        
        # Create negative pairs
        for i in range(min(1000, len(self.data) // 2)):
            quote1 = random.choice(self.data)
            quote2 = random.choice(self.data)
            
            # Ensure different authors or topics
            if quote1['author'] != quote2['author']:
                query = f"quotes by {quote1['author']}"
                examples.append(InputExample(texts=[query, quote2['quote']], label=0.0))
        
        logger.info(f"Created {len(examples)} training examples")
        return examples
    
    def fine_tune_model(self, output_path: str, epochs: int = 3):
        """Fine-tune the sentence transformer model"""
        logger.info(f"Loading base model: {self.base_model_name}")
        self.model = SentenceTransformer(self.base_model_name)
        
        # Create training examples
        train_examples = self.create_training_examples()
        
        # Create DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Fine-tune the model
        logger.info("Starting fine-tuning...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            output_path=output_path
        )
        
        logger.info(f"Model fine-tuned and saved to {output_path}")
        
    def evaluate_model(self, test_queries: List[str] = None):
        """Evaluate the fine-tuned model"""
        if not self.model:
            logger.error("Model not loaded")
            return
        
        if not test_queries:
            test_queries = [
                "quotes about love",
                "Shakespeare quotes",
                "motivational quotes",
                "quotes about success"
            ]
        
        logger.info("Evaluating model with test queries...")
        
        # Get all quote texts for similarity comparison
        quote_texts = [
            f"{item['quote']} - {item['author']} (Tags: {', '.join(item.get('tags', []))})"
            for item in self.data
        ]
        quote_embeddings = self.model.encode(quote_texts)
        
        for query in test_queries:
            query_embedding = self.model.encode([query])
            
            # Calculate similarities
            similarities = torch.cosine_similarity(
                torch.tensor(query_embedding), 
                torch.tensor(quote_embeddings)
            )
            
            # Get top 3 results
            top_indices = similarities.argsort(descending=True)[:3]
            
            print(f"\nQuery: '{query}'")
            print("Top results:")
            for i, idx in enumerate(top_indices):
                quote_data = self.data[idx]
                print(f"{i+1}. {quote_data['quote'][:100]}... - {quote_data['author']} "
                      f"(Score: {similarities[idx]:.3f})")

def main():
    # Initialize trainer
    trainer = QuoteEmbeddingTrainer()
    
    # Load data
    trainer.load_processed_data('data/processed_quotes.json')
    
    # Fine-tune model
    trainer.fine_tune_model('models/fine_tuned_model', epochs=3)
    
    # Load the fine-tuned model for evaluation
    trainer.model = SentenceTransformer('models/fine_tuned_model')
    
    # Evaluate model
    trainer.evaluate_model()

if __name__ == "__main__":
    main()