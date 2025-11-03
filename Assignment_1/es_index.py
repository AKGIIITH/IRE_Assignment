from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import time
from typing import Iterable, Tuple
from index_base import IndexBase

class MyElasticsearchIndex(IndexBase):
    """Elasticsearch-based search index."""
    
    def __init__(self, core='ESIndex', info='BOOLEAN', dstore='DB1', 
                 qproc='TERMatat', compr='NONE', optim='Null', 
                 es_host='localhost', es_port=9200):
        super().__init__(core, info, dstore, qproc, compr, optim)
        
        # Connect to Elasticsearch with retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Initialize Elasticsearch client
                self.es = Elasticsearch(
                    [f'http://{es_host}:{es_port}'],
                    request_timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )
                
                # Test connection with info() instead of ping()
                info = self.es.info()
                print(f"Connected to Elasticsearch at {es_host}:{es_port}")
                print(f"Cluster: {info['cluster_name']}, Version: {info['version']['number']}")
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Connection attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(
                        f"Cannot connect to Elasticsearch at {es_host}:{es_port}. "
                        f"Make sure it's running. Error: {e}"
                    )
    
    def create_index(self, index_id: str, files: Iterable[Tuple[str, str]]) -> None:
        """Create Elasticsearch index and index documents."""
        print(f"Creating Elasticsearch index: {index_id}")
        
        # Define index settings and mappings
        settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "english_analyzer": {
                            "type": "english"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "english_analyzer"
                    }
                }
            }
        }
        
        # Delete index if exists
        if self.es.indices.exists(index=index_id):
            self.es.indices.delete(index=index_id)
        
        # Create index
        self.es.indices.create(index=index_id, body=settings)
        
        # Prepare documents for bulk indexing
        def generate_docs():
            for doc_id, content in files:
                yield {
                    "_index": index_id,
                    "_id": doc_id,
                    "_source": {
                        "doc_id": doc_id,
                        "content": content
                    }
                }
        
        # Bulk index documents
        success, failed = bulk(self.es, generate_docs(), raise_on_error=False)
        
        # Refresh index to make documents searchable
        self.es.indices.refresh(index=index_id)
        
        print(f"Indexed {success} documents successfully")
        if failed:
            print(f"Failed to index {len(failed)} documents")
    
    def load_index(self, serialized_index_dump: str) -> None:
        """Load index - not needed for ES as it's always available."""
        print("Elasticsearch index is always loaded")
    
    def query(self, query: str, index_id: str = "default_index") -> str:
        """Execute query against Elasticsearch."""
        # Simple query implementation
        # Remove quotes for ES query
        clean_query = query.replace('"', '')
        
        # Build ES query
        es_query = {
            "query": {
                "match": {
                    "content": clean_query
                }
            },
            "size": 100
        }
        
        try:
            response = self.es.search(index=index_id, body=es_query)
            
            # Format results
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'doc_id': hit['_source']['doc_id'],
                    'score': hit['_score']
                })
            
            return json.dumps(results, indent=2)
        
        except Exception as e:
            print(f"Query error: {e}")
            return json.dumps([])
    
    def update_index(self, index_id: str, remove_files: Iterable[Tuple[str, str]], 
                    add_files: Iterable[Tuple[str, str]]) -> None:
        """Update Elasticsearch index."""
        # Remove documents
        for doc_id, _ in remove_files:
            try:
                self.es.delete(index=index_id, id=doc_id)
            except:
                pass
        
        # Add documents
        def generate_docs():
            for doc_id, content in add_files:
                yield {
                    "_index": index_id,
                    "_id": doc_id,
                    "_source": {
                        "doc_id": doc_id,
                        "content": content
                    }
                }
        
        bulk(self.es, generate_docs(), raise_on_error=False)
        self.es.indices.refresh(index=index_id)
    
    def delete_index(self, index_id: str) -> None:
        """Delete Elasticsearch index."""
        if self.es.indices.exists(index=index_id):
            self.es.indices.delete(index=index_id)
            print(f"Deleted index: {index_id}")
    
    def list_indices(self) -> Iterable[str]:
        """List all Elasticsearch indices."""
        indices = self.es.cat.indices(format='json')
        return [idx['index'] for idx in indices if not idx['index'].startswith('.')]
    
    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        """List all document IDs in index."""
        try:
            response = self.es.search(
                index=index_id,
                body={
                    "query": {"match_all": {}},
                    "_source": ["doc_id"],
                    "size": 10000
                }
            )
            
            return [hit['_source']['doc_id'] for hit in response['hits']['hits']]
        except:
            return []