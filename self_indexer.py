"""
Custom indexer (SelfIndex-v1.0) for IRE assignment.
Implements boolean indexing, inverted index, TF-IDF scoring, and persistence.
"""

import os
import json
import pickle
import math
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
from preprocessor import TextPreprocessor
from query_parser import QueryParser, QueryExecutor

class SelfIndexer:
    def __init__(self, index_name='selfindex-v1.0', storage_path='./index_storage'):
        """
        Initialize custom indexer.
        
        Args:
            index_name (str): Name of the index
            storage_path (str): Directory to store index files
        """
        self.index_name = index_name
        self.storage_path = storage_path
        self.preprocessor = TextPreprocessor()
        self.query_parser = QueryParser()
        self.query_executor = QueryExecutor(self)
        
        # Index data structures
        self.inverted_index = defaultdict(set)  # term -> set of doc_ids
        self.positional_index = defaultdict(dict)  # term -> {doc_id: [positions]}
        self.document_store = {}  # doc_id -> document content
        self.document_terms = {}  # doc_id -> list of terms
        self.term_frequencies = defaultdict(dict)  # term -> {doc_id: frequency}
        self.document_frequencies = defaultdict(int)  # term -> number of docs containing term
        self.document_lengths = {}  # doc_id -> number of terms
        self.total_documents = 0
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing index if available
        self.load_index()
    
    def add_document(self, doc_id: str, title: str, content: str):
        """
        Add a document to the index.
        
        Args:
            doc_id (str): Document ID
            title (str): Document title
            content (str): Document content
        """
        # Store document
        self.document_store[doc_id] = {
            'title': title,
            'content': content
        }
        
        # Preprocess text
        full_text = f"{title} {content}"
        terms = self.preprocessor.preprocess_text(full_text)
        
        # Store document terms
        self.document_terms[doc_id] = terms
        self.document_lengths[doc_id] = len(terms)
        
        # Build inverted index and positional index
        term_counts = Counter(terms)
        
        for position, term in enumerate(terms):
            # Add to inverted index
            self.inverted_index[term].add(doc_id)
            
            # Add to positional index
            if doc_id not in self.positional_index[term]:
                self.positional_index[term][doc_id] = []
            self.positional_index[term][doc_id].append(position)
            
            # Update term frequencies
            self.term_frequencies[term][doc_id] = term_counts[term]
        
        # Update document frequencies
        for term in set(terms):
            self.document_frequencies[term] += 1
        
        self.total_documents += 1
    
    def search_term(self, term: str) -> Set[str]:
        """
        Search for documents containing a term.
        
        Args:
            term (str): Search term
            
        Returns:
            Set[str]: Set of document IDs
        """
        # Preprocess the search term
        processed_terms = self.preprocessor.preprocess_text(term)
        if not processed_terms:
            return set()
        
        processed_term = processed_terms[0]
        return self.inverted_index.get(processed_term, set())
    
    def search_phrase(self, phrase: str) -> Set[str]:
        """
        Search for documents containing a phrase.
        
        Args:
            phrase (str): Search phrase
            
        Returns:
            Set[str]: Set of document IDs
        """
        # Preprocess the phrase
        terms = self.preprocessor.preprocess_text(phrase)
        if not terms:
            return set()
        
        if len(terms) == 1:
            return self.search_term(terms[0])
        
        # Find documents containing all terms
        candidate_docs = self.inverted_index.get(terms[0], set())
        for term in terms[1:]:
            candidate_docs = candidate_docs.intersection(
                self.inverted_index.get(term, set())
            )
        
        # Check for phrase matches using positional index
        phrase_matches = set()
        
        for doc_id in candidate_docs:
            # Get positions for each term
            positions = []
            for term in terms:
                if term in self.positional_index and doc_id in self.positional_index[term]:
                    positions.append(self.positional_index[term][doc_id])
                else:
                    break
            
            if len(positions) == len(terms):
                # Check if terms appear consecutively
                if self._check_phrase_match(positions):
                    phrase_matches.add(doc_id)
        
        return phrase_matches
    
    def _check_phrase_match(self, term_positions: List[List[int]]) -> bool:
        """
        Check if terms appear as a consecutive phrase.
        
        Args:
            term_positions (List[List[int]]): Positions for each term
            
        Returns:
            bool: True if phrase match found
        """
        if not term_positions:
            return False
        
        # Check all combinations of positions
        for pos1 in term_positions[0]:
            current_pos = pos1
            match = True
            
            for i in range(1, len(term_positions)):
                expected_pos = current_pos + 1
                if expected_pos not in term_positions[i]:
                    match = False
                    break
                current_pos = expected_pos
            
            if match:
                return True
        
        return False
    
    def calculate_tf_idf(self, term: str, doc_id: str) -> float:
        """
        Calculate TF-IDF score for a term in a document.
        
        Args:
            term (str): Term
            doc_id (str): Document ID
            
        Returns:
            float: TF-IDF score
        """
        if term not in self.term_frequencies or doc_id not in self.term_frequencies[term]:
            return 0.0
        
        # Term frequency (normalized)
        tf = self.term_frequencies[term][doc_id] / self.document_lengths[doc_id]
        
        # Inverse document frequency
        df = self.document_frequencies[term]
        if df == 0:
            return 0.0
        
        idf = math.log(self.total_documents / df)
        
        return tf * idf
    
    def search_with_ranking(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search with TF-IDF ranking.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[str, float]]: List of (doc_id, score) tuples
        """
        # Preprocess query
        query_terms = self.preprocessor.preprocess_text(query)
        if not query_terms:
            return []
        
        # Get candidate documents
        candidate_docs = set()
        for term in query_terms:
            candidate_docs.update(self.inverted_index.get(term, set()))
        
        # Calculate scores
        doc_scores = {}
        for doc_id in candidate_docs:
            score = 0.0
            for term in query_terms:
                score += self.calculate_tf_idf(term, doc_id)
            doc_scores[doc_id] = score
        
        # Sort by score and return top-k
        ranked_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_results[:top_k]
    
    def boolean_search(self, query: str) -> Set[str]:
        """
        Perform boolean search using query parser.
        
        Args:
            query (str): Boolean query
            
        Returns:
            Set[str]: Set of document IDs
        """
        try:
            ast = self.query_parser.parse(query)
            return self.query_executor.execute(ast)
        except Exception as e:
            print(f"Error in boolean search: {e}")
            return set()
    
    def get_all_document_ids(self) -> Set[str]:
        """Get all document IDs in the index."""
        return set(self.document_store.keys())
    
    def get_document(self, doc_id: str) -> Dict:
        """Get document by ID."""
        return self.document_store.get(doc_id, {})
    
    def get_index_stats(self) -> Dict:
        """Get index statistics."""
        return {
            'total_documents': self.total_documents,
            'total_terms': len(self.inverted_index),
            'average_document_length': sum(self.document_lengths.values()) / max(1, len(self.document_lengths)),
            'index_size_mb': self._estimate_index_size() / (1024 * 1024)
        }
    
    def _estimate_index_size(self) -> int:
        """Estimate index size in bytes."""
        size = 0
        
        # Estimate inverted index size
        for term, doc_ids in self.inverted_index.items():
            size += len(term.encode('utf-8'))
            size += len(doc_ids) * 50  # Approximate doc_id size
        
        # Estimate positional index size
        for term, doc_positions in self.positional_index.items():
            for doc_id, positions in doc_positions.items():
                size += len(positions) * 4  # 4 bytes per position
        
        # Estimate document store size
        for doc_id, doc in self.document_store.items():
            size += len(json.dumps(doc).encode('utf-8'))
        
        return size
    
    def save_index(self):
        """Save index to disk."""
        index_data = {
            'inverted_index': dict(self.inverted_index),
            'positional_index': dict(self.positional_index),
            'document_store': self.document_store,
            'document_terms': self.document_terms,
            'term_frequencies': dict(self.term_frequencies),
            'document_frequencies': dict(self.document_frequencies),
            'document_lengths': self.document_lengths,
            'total_documents': self.total_documents
        }
        
        file_path = os.path.join(self.storage_path, f'{self.index_name}.pkl')
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(index_data, f)
            print(f"Index saved to {file_path}")
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self):
        """Load index from disk."""
        file_path = os.path.join(self.storage_path, f'{self.index_name}.pkl')
        
        if not os.path.exists(file_path):
            print(f"No existing index found at {file_path}")
            return
        
        try:
            with open(file_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.inverted_index = defaultdict(set, {k: set(v) for k, v in index_data['inverted_index'].items()})
            self.positional_index = defaultdict(dict, index_data['positional_index'])
            self.document_store = index_data['document_store']
            self.document_terms = index_data['document_terms']
            self.term_frequencies = defaultdict(dict, index_data['term_frequencies'])
            self.document_frequencies = defaultdict(int, index_data['document_frequencies'])
            self.document_lengths = index_data['document_lengths']
            self.total_documents = index_data['total_documents']
            
            print(f"Index loaded from {file_path}")
            print(f"Loaded {self.total_documents} documents")
        except Exception as e:
            print(f"Error loading index: {e}")

if __name__ == "__main__":
    # Example usage
    indexer = SelfIndexer()
    
    # Sample documents
    sample_docs = [
        {
            "id": "doc1",
            "title": "Information Retrieval",
            "content": "Information retrieval is the activity of obtaining information system resources that are relevant to an information need from a collection of those resources."
        },
        {
            "id": "doc2",
            "title": "Search Engines",
            "content": "Search engines are complex distributed systems that help users find information. They use sophisticated algorithms for indexing and ranking."
        },
        {
            "id": "doc3",
            "title": "Boolean Logic",
            "content": "Boolean logic uses operators like AND, OR, and NOT to combine search terms. This allows for precise query formulation."
        }
    ]
    
    # Add documents
    print("Adding documents to index...")
    for doc in sample_docs:
        indexer.add_document(doc['id'], doc['title'], doc['content'])
    
    # Save index
    indexer.save_index()
    
    # Search examples
    print("\n--- Boolean Search Examples ---")
    
    test_queries = [
        "information",
        "information AND retrieval",
        "search OR boolean",
        "NOT algorithms",
        '"information retrieval"'
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = indexer.boolean_search(query)
        print(f"Results: {list(results)}")
        
        for doc_id in list(results)[:3]:  # Show first 3 results
            doc = indexer.get_document(doc_id)
            print(f"  {doc_id}: {doc.get('title', 'No title')}")
    
    # TF-IDF ranking example
    print("\n--- TF-IDF Ranking Example ---")
    query = "information search"
    ranked_results = indexer.search_with_ranking(query)
    
    print(f"Query: {query}")
    for doc_id, score in ranked_results:
        doc = indexer.get_document(doc_id)
        print(f"  {doc_id} (score: {score:.4f}): {doc.get('title', 'No title')}")
    
    # Index statistics
    print("\n--- Index Statistics ---")
    stats = indexer.get_index_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
