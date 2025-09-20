"""
Elasticsearch indexer (ESIndex-v1.0) for IRE assignment.
Handles document indexing and querying using Elasticsearch.
"""

from elasticsearch import Elasticsearch
import json
import time
from preprocessor import TextPreprocessor

class ESIndexer:
    def __init__(self, host='localhost', port=9200, index_name='esindex-v1.0'):
        """
        Initialize Elasticsearch indexer.
        
        Args:
            host (str): Elasticsearch host
            port (int): Elasticsearch port
            index_name (str): Name of the index
        """
        self.host = host
        self.port = port
        self.index_name = index_name
        self.preprocessor = TextPreprocessor()
        
        # Initialize Elasticsearch client
        try:
            self.es = Elasticsearch([{'host': host, 'port': port}])
            print(f"Connected to Elasticsearch at {host}:{port}")
        except Exception as e:
            print(f"Warning: Could not connect to Elasticsearch: {e}")
            print("Make sure Elasticsearch is running on localhost:9200")
            self.es = None
    
    def create_index(self):
        """Create the index with proper mappings."""
        if not self.es:
            print("Elasticsearch not connected")
            return False
        
        # Define index mapping
        mapping = {
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "doc_id": {
                        "type": "keyword"
                    },
                    "processed_content": {
                        "type": "text",
                        "analyzer": "keyword"
                    }
                }
            }
        }
        
        try:
            if self.es.indices.exists(index=self.index_name):
                print(f"Index {self.index_name} already exists")
                return True
            
            self.es.indices.create(index=self.index_name, body=mapping)
            print(f"Created index: {self.index_name}")
            return True
        except Exception as e:
            print(f"Error creating index: {e}")
            return False
    
    def index_document(self, doc_id, title, content):
        """
        Index a single document.
        
        Args:
            doc_id (str): Document ID
            title (str): Document title
            content (str): Document content
        """
        if not self.es:
            print("Elasticsearch not connected")
            return False
        
        # Preprocess content
        processed_tokens = self.preprocessor.preprocess_text(content)
        processed_content = ' '.join(processed_tokens)
        
        document = {
            "doc_id": doc_id,
            "title": title,
            "content": content,
            "processed_content": processed_content
        }
        
        try:
            self.es.index(index=self.index_name, id=doc_id, body=document)
            return True
        except Exception as e:
            print(f"Error indexing document {doc_id}: {e}")
            return False
    
    def index_documents(self, documents):
        """
        Index multiple documents.
        
        Args:
            documents (list): List of dictionaries with 'id', 'title', 'content'
        """
        if not self.es:
            print("Elasticsearch not connected")
            return
        
        success_count = 0
        for doc in documents:
            if self.index_document(doc['id'], doc['title'], doc['content']):
                success_count += 1
        
        print(f"Successfully indexed {success_count}/{len(documents)} documents")
        
        # Refresh index to make documents searchable
        self.es.indices.refresh(index=self.index_name)
    
    def search(self, query, field='content', size=10):
        """
        Search documents.
        
        Args:
            query (str): Search query
            field (str): Field to search in
            size (int): Number of results to return
            
        Returns:
            dict: Search results
        """
        if not self.es:
            print("Elasticsearch not connected")
            return {'hits': {'hits': []}}
        
        search_body = {
            "query": {
                "match": {
                    field: query
                }
            },
            "size": size
        }
        
        try:
            response = self.es.search(index=self.index_name, body=search_body)
            return response
        except Exception as e:
            print(f"Error searching: {e}")
            return {'hits': {'hits': []}}
    
    def boolean_search(self, query_dict):
        """
        Perform boolean search.
        
        Args:
            query_dict (dict): Boolean query structure
            
        Returns:
            dict: Search results
        """
        if not self.es:
            print("Elasticsearch not connected")
            return {'hits': {'hits': []}}
        
        search_body = {
            "query": {
                "bool": query_dict
            }
        }
        
        try:
            response = self.es.search(index=self.index_name, body=search_body)
            return response
        except Exception as e:
            print(f"Error in boolean search: {e}")
            return {'hits': {'hits': []}}
    
    def get_document_count(self):
        """Get total number of documents in the index."""
        if not self.es:
            return 0
        
        try:
            response = self.es.count(index=self.index_name)
            return response['count']
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    def delete_index(self):
        """Delete the index."""
        if not self.es:
            print("Elasticsearch not connected")
            return False
        
        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                print(f"Deleted index: {self.index_name}")
                return True
            else:
                print(f"Index {self.index_name} does not exist")
                return False
        except Exception as e:
            print(f"Error deleting index: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    indexer = ESIndexer()
    
    # Create index
    indexer.create_index()
    
    # Sample documents
    sample_docs = [
        {
            "id": "doc1",
            "title": "Information Retrieval",
            "content": "Information retrieval is the activity of obtaining information system resources that are relevant to an information need from a collection of those resources."
        },
        {
            "id": "doc2",
            "title": "Elasticsearch Guide",
            "content": "Elasticsearch is a search engine based on the Lucene library. It provides a distributed, multitenant-capable full-text search engine."
        },
        {
            "id": "doc3",
            "title": "Boolean Search",
            "content": "Boolean search is a type of search allowing users to combine keywords with operators such as AND, OR, and NOT to further produce more relevant results."
        }
    ]
    
    # Index documents
    indexer.index_documents(sample_docs)
    
    # Search examples
    print("\nSearch Results:")
    results = indexer.search("information retrieval")
    for hit in results['hits']['hits']:
        print(f"Doc ID: {hit['_source']['doc_id']}")
        print(f"Title: {hit['_source']['title']}")
        print(f"Score: {hit['_score']}")
        print()
    
    print(f"Total documents in index: {indexer.get_document_count()}")
