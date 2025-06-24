import json
import os
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_recall,
    context_precision,
)
from datasets import Dataset
from typing import List, Dict
import logging
from rag_pipeline import RAGQuoteSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, rag_system: RAGQuoteSystem):
        self.rag_system = rag_system
        self.test_queries = []
        self.ground_truth = []
        
    def create_evaluation_dataset(self) -> List[Dict]:
        """Create evaluation dataset with test queries and expected results"""
        evaluation_data = [
            {
                "question": "quotes about love and relationships",
                "expected_authors": ["Shakespeare", "Jane Austen", "Pablo Neruda"],
                "expected_themes": ["love", "romance", "relationships"]
            },
            {
                "question": "motivational quotes for success",
                "expected_authors": ["Napoleon Hill", "Dale Carnegie", "Tony Robbins"],
                "expected_themes": ["success", "motivation", "achievement"]
            },
            {
                "question": "quotes about courage and bravery",
                "expected_authors": ["Winston Churchill", "Eleanor Roosevelt", "Maya Angelou"],
                "expected_themes": ["courage", "bravery", "strength"]
            },
            {
                "question": "philosophical quotes about life",
                "expected_authors": ["Aristotle", "Socrates", "Nietzsche"],
                "expected_themes": ["philosophy", "life", "wisdom"]
            },
            {
                "question": "quotes about friendship",
                "expected_authors": ["C.S. Lewis", "Ralph Waldo Emerson", "Aristotle"],
                "expected_themes": ["friendship", "companionship", "loyalty"]
            }
        ]
        
        return evaluation_data
    
    def generate_rag_responses(self, test_data: List[Dict]) -> List[Dict]:
        """Generate RAG responses for evaluation"""
        responses = []
        
        for item in test_data:
            query = item["question"]
            
            # Get RAG response
            rag_result = self.rag_system.search(query, top_k=5)
            
            # Extract contexts (retrieved quotes)
            contexts = []
            for quote in rag_result['retrieved_quotes']:
                context = f"\"{quote['quote']}\" - {quote['author']}"
                if quote['tags']:
                    context += f" (Tags: {', '.join(quote['tags'])})"
                contexts.append(context)
            
            # Prepare response data
            response_data = {
                "question": query,
                "answer": rag_result.get('llm_response', 'No LLM response generated'),
                "contexts": contexts,
                "ground_truths": [f"Expected themes: {', '.join(item['expected_themes'])}"],
                "expected_authors": item["expected_authors"],
                "expected_themes": item["expected_themes"]
            }
            
            responses.append(response_data)
        
        return responses
    
    def evaluate_with_ragas(self, responses: List[Dict]) -> Dict:
        """Evaluate RAG system using RAGAS metrics"""
        # Prepare dataset for RAGAS
        eval_dataset = Dataset.from_list([
            {
                "question": resp["question"],
                "answer": resp["answer"],
                "contexts": resp["contexts"],
                "ground_truths": resp["ground_truths"]
            }
            for resp in responses
        ])
        
        # Define metrics to evaluate
        metrics = [
            context_precision,
            context_recall,
            faithfulness
        ]
        
        try:
            # Run evaluation
            logger.info("Running RAGAS evaluation...")
            result = evaluate(eval_dataset, metrics=metrics)
            
            return result
        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {e}")
            return None
    
    def custom_evaluation_metrics(self, responses: List[Dict]) -> Dict:
        """Custom evaluation metrics specific to quote retrieval"""
        metrics = {
            "author_accuracy": 0,
            "theme_coverage": 0,
            "avg_similarity_score": 0,
            "response_quality": 0
        }
        
        total_queries = len(responses)
        author_matches = 0
        theme_matches = 0
        total_similarity = 0
        
        for response in responses:
            # Check author accuracy
            retrieved_authors = []
            for quote in response.get('contexts', []):
                # Extract author from context string
                if ' - ' in quote:
                    author = quote.split(' - ')[1].split(' (')[0]
                    retrieved_authors.append(author)
            
            expected_authors = response['expected_authors']
            author_overlap = len(set(retrieved_authors) & set(expected_authors))
            if author_overlap > 0:
                author_matches += 1
            
            response_text = response['answer'].lower()
            expected_themes = response['expected_themes']
            theme_overlap = sum(1 for theme in expected_themes if theme.lower() in response_text)
            if theme_overlap > 0:
                theme_matches += 1
        
        metrics["author_accuracy"] = author_matches / total_queries
        metrics["theme_coverage"] = theme_matches / total_queries
        
        return metrics
    
    def run_complete_evaluation(self) -> Dict:
        """Run complete evaluation of the RAG system"""
        logger.info("Starting RAG evaluation...")
        
        # Create evaluation dataset
        test_data = self.create_evaluation_dataset()
        
        # Generate RAG responses
        responses = self.generate_rag_responses(test_data)
        
        # Run RAGAS evaluation
        ragas_results = self.evaluate_with_ragas(responses)
        
        # Run custom evaluation
        custom_results = self.custom_evaluation_metrics(responses)
        
        # Combine results
        evaluation_results = {
            "ragas_metrics": ragas_results.to_dict() if ragas_results else {},
            "custom_metrics": custom_results,
            "sample_responses": responses[:2]  
        }
        
        return evaluation_results
    
    def save_evaluation_report(self, results: Dict, filepath: str):
        """Save evaluation results to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation report saved to {filepath}")

def main():
    # Initialize RAG system
    rag_system = RAGQuoteSystem(
        model_path='models/fine_tuned_model',
        openai_api_key=os.getenv('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')
    )
    
    # Load model and data
    rag_system.load_model_and_data('data/processed_quotes.json')
    rag_system.load_index('models/quote_index.faiss')
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_system)
    
    # Run evaluation
    results = evaluator.run_complete_evaluation()
    
    # Print results
    print("\n" + "="*50)
    print("RAG EVALUATION RESULTS")
    print("="*50)
    
    if results["ragas_metrics"]:
        print("\nRAGAS Metrics:")
        for metric, value in results["ragas_metrics"].items():
            print(f"  {metric}: {value:.3f}")
    
    print("\nCustom Metrics:")
    for metric, value in results["custom_metrics"].items():
        print(f"  {metric}: {value:.3f}")
    
    # Save results
    evaluator.save_evaluation_report(results, 'evaluation_report.json')

if __name__ == "__main__":
    main()