import pandas as pd
from datasets import load_dataset
import json
import re
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuoteDataPreprocessor:
    def __init__(self):
        self.dataset = None
        self.processed_data = []
        
    def load_dataset(self, local_path: str = "data/processed_quotes.json"):
        """Load the quotes dataset from a local JSON file"""
        logger.info("Loading dataset from local file...")
        with open(local_path, "r", encoding="utf-8") as f:
            self.processed_data = json.load(f)
        logger.info(f"Loaded {len(self.processed_data)} quotes from local file.")
        return self.processed_data

    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:\'"()-]', '', text)
        
        return text
    
    def preprocess_quotes(self) -> List[Dict]:
        """Preprocess the quotes dataset"""
        logger.info("Preprocessing quotes...")
        
        for idx, item in enumerate(self.dataset['train']):
            # Extract and clean fields
            quote = self.clean_text(item.get('quote', ''))
            author = self.clean_text(item.get('author', 'Unknown'))
            tags = item.get('tags', [])
            
            # Skip empty quotes
            if not quote or len(quote.strip()) < 10:
                continue
                
            # Process tags
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
            elif not isinstance(tags, list):
                tags = []
            
            # Create processed entry
            processed_entry = {
                'id': idx,
                'quote': quote,
                'author': author,
                'tags': tags,
                'combined_text': f"{quote} - {author} {' '.join(tags)}"
            }
            
            self.processed_data.append(processed_entry)
        
        logger.info(f"Processed {len(self.processed_data)} quotes")
        return self.processed_data
    
    def save_processed_data(self, filepath: str):
        """Save processed data to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Processed data saved to {filepath}")
    
    def get_data_statistics(self):
        """Get basic statistics about the dataset"""
        if not self.processed_data:
            return None
            
        authors = [item['author'] for item in self.processed_data]
        all_tags = []
        for item in self.processed_data:
            all_tags.extend(item['tags'])
        
        stats = {
            'total_quotes': len(self.processed_data),
            'unique_authors': len(set(authors)),
            'most_common_authors': pd.Series(authors).value_counts().head(10).to_dict(),
            'total_tags': len(all_tags),
            'unique_tags': len(set(all_tags)),
            'most_common_tags': pd.Series(all_tags).value_counts().head(10).to_dict()
        }
        
        return stats

def main():
    preprocessor = QuoteDataPreprocessor()

    # Load data from the local processed JSON file
    preprocessor.load_dataset("data/processed_quotes.json")
    
    # No need to preprocess or save again unless you want to
    stats = preprocessor.get_data_statistics()

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total quotes: {stats['total_quotes']}")
    print(f"Unique authors: {stats['unique_authors']}")
    print(f"Unique tags: {stats['unique_tags']}")
    print("\nTop authors:", stats['most_common_authors'])
    print("\nTop tags:", stats['most_common_tags'])

if __name__ == "__main__":
    main()