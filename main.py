"""
Main application for IRE Assignment.
Demonstrates both Elasticsearch (ESIndex-v1.0) and Custom (SelfIndex-v1.0) indexing systems.
"""

import time
import os
from typing import List, Dict
import matplotlib.pyplot as plt

from preprocessor import TextPreprocessor
from es_indexer import ESIndexer
from self_indexer import SelfIndexer

class IRESampleData:
    """Generate sample data for testing."""
    
    @staticmethod
    def get_sample_documents() -> List[Dict]:
        """Get sample documents for indexing."""
        return [
            {
                "id": "doc1",
                "title": "Introduction to Information Retrieval",
                "content": "Information retrieval is a field concerned with the structure, analysis, organization, storage, searching, and retrieval of information. Modern information retrieval systems use sophisticated algorithms to index and rank documents based on relevance to user queries."
            },
            {
                "id": "doc2",
                "title": "Search Engine Architecture",
                "content": "Search engines consist of several components including web crawlers, indexing systems, query processors, and ranking algorithms. The indexing phase involves building inverted indexes that map terms to documents for efficient retrieval."
            },
            {
                "id": "doc3",
                "title": "Boolean Search Models",
                "content": "Boolean search allows users to combine keywords using logical operators such as AND, OR, and NOT. This precise query formulation enables users to specify exact information needs and retrieve highly relevant documents from large collections."
            },
            {
                "id": "doc4",
                "title": "Vector Space Model",
                "content": "The vector space model represents documents and queries as vectors in a high-dimensional space. Document similarity is computed using cosine similarity, and TF-IDF weighting schemes help determine term importance within documents and across the collection."
            },
            {
                "id": "doc5",
                "title": "Text Preprocessing Techniques",
                "content": "Text preprocessing involves several steps including tokenization, stopword removal, stemming, and normalization. These techniques help reduce vocabulary size and improve matching between queries and documents by normalizing textual variations."
            },
            {
                "id": "doc6",
                "title": "Elasticsearch and Lucene",
                "content": "Elasticsearch is a distributed search engine built on Apache Lucene. It provides real-time search capabilities, horizontal scalability, and advanced query features including full-text search, aggregations, and complex boolean queries."
            },
            {
                "id": "doc7",
                "title": "Inverted Index Structure",
                "content": "An inverted index is a data structure that maps each unique term to a list of documents containing that term. This structure enables efficient term-based searches and is fundamental to most modern search engines and information retrieval systems."
            },
            {
                "id": "doc8",
                "title": "Query Processing Pipeline",
                "content": "Query processing involves parsing user queries, applying preprocessing transformations, retrieving candidate documents from indexes, computing relevance scores, and ranking results. Advanced systems support complex query types including phrase queries and proximity searches."
            },
            {
                "id": "doc9",
                "title": "Evaluation Metrics in IR",
                "content": "Information retrieval systems are evaluated using metrics such as precision, recall, F-measure, and mean average precision. These metrics help assess system effectiveness in retrieving relevant documents and ranking them appropriately for user needs."
            },
            {
                "id": "doc10",
                "title": "Machine Learning in Search",
                "content": "Modern search systems increasingly use machine learning techniques for ranking, query understanding, and personalization. Neural networks, deep learning models, and large language models are transforming how search engines understand and match user information needs."
            }
        ]
    
    @staticmethod
    def get_test_queries() -> List[str]:
        """Get test queries for evaluation."""
        return [
            "information retrieval",
            "search engine",
            "boolean AND query",
            "vector space model",
            '"text preprocessing"',
            "elasticsearch OR lucene",
            "NOT machine learning",
            "(information AND retrieval) OR (search AND engine)",
            "inverted index structure",
            "evaluation metrics precision recall"
        ]

class PerformanceEvaluator:
    """Evaluate system performance metrics."""
    
    def __init__(self):
        self.results = {}
    
    def measure_latency(self, indexer, queries: List[str], search_method='boolean') -> Dict:
        """
        Measure search latency for a set of queries.
        
        Args:
            indexer: Indexer instance
            queries: List of test queries
            search_method: 'boolean' or 'ranked'
            
        Returns:
            Dict: Latency statistics
        """
        latencies = []
        
        for query in queries:
            start_time = time.time()
            
            try:
                if search_method == 'boolean':
                    if hasattr(indexer, 'boolean_search'):
                        results = indexer.boolean_search(query)
                    else:
                        results = indexer.search(query)
                else:  # ranked search
                    if hasattr(indexer, 'search_with_ranking'):
                        results = indexer.search_with_ranking(query)
                    else:
                        results = indexer.search(query)
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
                
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                latencies.append(0)
        
        # Calculate percentiles
        latencies.sort()
        n = len(latencies)
        
        stats = {
            'mean': sum(latencies) / n if n > 0 else 0,
            'median': latencies[n // 2] if n > 0 else 0,
            'p95': latencies[int(0.95 * n)] if n > 0 else 0,
            'p99': latencies[int(0.99 * n)] if n > 0 else 0,
            'min': min(latencies) if latencies else 0,
            'max': max(latencies) if latencies else 0,
            'total_queries': n
        }
        
        return stats
    
    def estimate_memory_footprint(self, indexer) -> Dict:
        """Estimate memory footprint of the indexer."""
        if hasattr(indexer, 'get_index_stats'):
            stats = indexer.get_index_stats()
            return {
                'estimated_size_mb': stats.get('index_size_mb', 0),
                'total_documents': stats.get('total_documents', 0),
                'total_terms': stats.get('total_terms', 0)
            }
        else:
            return {
                'estimated_size_mb': 'N/A',
                'total_documents': 'N/A',
                'total_terms': 'N/A'
            }

def main():
    """Main application function."""
    print("=== IRE Assignment Demo ===")
    print("Implementing ESIndex-v1.0 and SelfIndex-v1.0\n")
    
    # Initialize components
    preprocessor = TextPreprocessor()
    evaluator = PerformanceEvaluator()
    
    # Get sample data
    sample_docs = IRESampleData.get_sample_documents()
    test_queries = IRESampleData.get_test_queries()
    
    print(f"Sample dataset: {len(sample_docs)} documents")
    print(f"Test queries: {len(test_queries)} queries\n")
    
    # Generate word frequency plots
    print("=== Text Preprocessing Analysis ===")
    texts = [doc['content'] for doc in sample_docs]
    
    try:
        freq_raw, freq_processed = preprocessor.plot_word_frequencies(
            texts, top_n=15, save_path='word_frequencies.png'
        )
        print(f"Generated word frequency plots (saved as word_frequencies.png)")
        print(f"Vocabulary size - Raw: {len(freq_raw)}, Processed: {len(freq_processed)}\\n")
    except Exception as e:
        print(f"Could not generate plots: {e}\\n")
    
    # Initialize indexers
    print("=== Initializing Indexers ===")
    
    # Custom indexer (SelfIndex-v1.0)
    print("Initializing SelfIndex-v1.0...")
    self_indexer = SelfIndexer(index_name='selfindex-v1.0')
    
    # Elasticsearch indexer (ESIndex-v1.0)
    print("Initializing ESIndex-v1.0...")
    es_indexer = ESIndexer(index_name='esindex-v1.0')
    
    # Index documents
    print("\\n=== Indexing Documents ===")
    
    # Index with custom indexer
    print("Indexing with SelfIndex-v1.0...")
    start_time = time.time()
    for doc in sample_docs:
        self_indexer.add_document(doc['id'], doc['title'], doc['content'])
    self_indexer.save_index()
    self_indexing_time = time.time() - start_time
    print(f"SelfIndex-v1.0 indexing completed in {self_indexing_time:.3f} seconds")
    
    # Index with Elasticsearch (if available)
    print("Indexing with ESIndex-v1.0...")
    start_time = time.time()
    if es_indexer.es:
        es_indexer.create_index()
        es_indexer.index_documents(sample_docs)
        es_indexing_time = time.time() - start_time
        print(f"ESIndex-v1.0 indexing completed in {es_indexing_time:.3f} seconds")
    else:
        print("Elasticsearch not available - skipping ESIndex-v1.0")
        es_indexing_time = 0
    
    # Performance evaluation
    print("\\n=== Performance Evaluation ===")
    
    # Evaluate SelfIndex-v1.0
    print("Evaluating SelfIndex-v1.0...")
    self_latency = evaluator.measure_latency(self_indexer, test_queries, 'boolean')
    self_memory = evaluator.estimate_memory_footprint(self_indexer)
    
    print(f"SelfIndex-v1.0 Results:")
    print(f"  Mean latency: {self_latency['mean']:.2f} ms")
    print(f"  P95 latency: {self_latency['p95']:.2f} ms")
    print(f"  P99 latency: {self_latency['p99']:.2f} ms")
    print(f"  Memory footprint: {self_memory['estimated_size_mb']:.2f} MB")
    
    # Evaluate ESIndex-v1.0 (if available)
    if es_indexer.es:
        print("\\nEvaluating ESIndex-v1.0...")
        es_latency = evaluator.measure_latency(es_indexer, test_queries, 'boolean')
        es_memory = evaluator.estimate_memory_footprint(es_indexer)
        
        print(f"ESIndex-v1.0 Results:")
        print(f"  Mean latency: {es_latency['mean']:.2f} ms")
        print(f"  P95 latency: {es_latency['p95']:.2f} ms")
        print(f"  P99 latency: {es_latency['p99']:.2f} ms")
        print(f"  Memory footprint: {es_memory.get('estimated_size_mb', 'N/A')} MB")
    
    # Demonstrate search capabilities
    print("\\n=== Search Examples ===")
    
    demo_queries = [
        "information retrieval",
        "search AND engine",
        '"vector space model"',
        "elasticsearch OR lucene",
        "NOT machine learning"
    ]
    
    for query in demo_queries[:3]:  # Show first 3 queries
        print(f"\\nQuery: {query}")
        
        # SelfIndex search
        print("SelfIndex-v1.0 results:")
        try:
            results = self_indexer.boolean_search(query)
            for doc_id in list(results)[:3]:
                doc = self_indexer.get_document(doc_id)
                print(f"  {doc_id}: {doc.get('title', 'No title')}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # Elasticsearch search (if available)
        if es_indexer.es:
            print("ESIndex-v1.0 results:")
            try:
                results = es_indexer.search(query)
                for hit in results['hits']['hits'][:3]:
                    print(f"  {hit['_source']['doc_id']}: {hit['_source']['title']} (score: {hit['_score']:.2f})")
            except Exception as e:
                print(f"  Error: {e}")
    
    # Show TF-IDF ranking example (SelfIndex only)
    print("\\n=== TF-IDF Ranking Example (SelfIndex-v1.0) ===")
    ranking_query = "information retrieval systems"
    print(f"Query: {ranking_query}")
    
    try:
        ranked_results = self_indexer.search_with_ranking(ranking_query, top_k=5)
        for doc_id, score in ranked_results:
            doc = self_indexer.get_document(doc_id)
            print(f"  {doc_id} (score: {score:.4f}): {doc.get('title', 'No title')}")
    except Exception as e:
        print(f"Error in ranking: {e}")
    
    # Show index statistics
    print("\\n=== Index Statistics ===")
    
    self_stats = self_indexer.get_index_stats()
    print("SelfIndex-v1.0:")
    for key, value in self_stats.items():
        print(f"  {key}: {value}")
    
    if es_indexer.es:
        doc_count = es_indexer.get_document_count()
        print(f"\\nESIndex-v1.0:")
        print(f"  total_documents: {doc_count}")
    
    print("\\n=== Demo Complete ===")
    print("Check the generated files:")
    print("- word_frequencies.png: Word frequency analysis")
    print("- index_storage/: Persistent index files")

if __name__ == "__main__":
    main()
