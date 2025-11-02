import time
import json
import tracemalloc
from pathlib import Path
from typing import List, Dict
import numpy as np
from data_loader import get_all_documents
from es_index import MyElasticsearchIndex
from self_index import MySelfIndex

class Benchmark:
    """Benchmark different index configurations."""
    
    def __init__(self, query_file: str = "diverse-queries.json"):
        self.query_file = query_file
        self.results = []
        
        # Load queries
        if Path(query_file).exists():
            with open(query_file, 'r') as f:
                self.queries = json.load(f)
        else:
            # Default queries if file doesn't exist
            self.queries = [
                '"machine learning"',
                '"artificial intelligence" AND "neural networks"',
                '"python" OR "java"',
                '"data" AND NOT "science"',
                '"deep learning"'
            ]
            print(f"Query file not found, using default queries")
    
    def collect_documents(self, max_docs: int = 1000):
        """Collect documents for indexing."""
        print(f"Collecting {max_docs} documents...")
        docs = []
        for doc_id, content in get_all_documents(use_wiki=True, use_news=False, max_wiki_docs=max_docs):
            docs.append((doc_id, content))
            if len(docs) >= max_docs:
                break
        print(f"Collected {len(docs)} documents")
        return docs
    
    def measure_index_creation(self, index_obj, index_id: str, docs: List):
        """Measure index creation time and size."""
        print(f"\n{'='*60}")
        print(f"Testing: {index_obj.identifier_short}")
        print(f"{'='*60}")
        
        start_time = time.time()
        index_obj.create_index(index_id, docs)
        creation_time = time.time() - start_time
        
        # Measure disk size
        disk_size = self._get_index_size(index_obj)
        
        print(f"Creation time: {creation_time:.2f}s")
        print(f"Disk size: {disk_size / (1024*1024):.2f} MB")
        
        return creation_time, disk_size
    
    def _get_index_size(self, index_obj) -> int:
        """Get index size on disk in bytes."""
        total_size = 0
        
        if hasattr(index_obj, 'base_dir'):
            base_dir = Path(index_obj.base_dir)
            if base_dir.exists():
                for file in base_dir.rglob('*'):
                    if file.is_file():
                        total_size += file.stat().st_size
        
        return total_size
    
    def measure_query_performance(self, index_obj, index_id: str):
        """Measure query latency and throughput."""
        print(f"Running {len(self.queries)} queries...")
        
        latencies = []
        
        # Start memory tracking
        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]
        
        start_time = time.time()
        
        for query in self.queries:
            query_start = time.time()
            
            try:
                if isinstance(index_obj, MyElasticsearchIndex):
                    results = index_obj.query(query, index_id)
                else:
                    results = index_obj.query(query)
            except Exception as e:
                print(f"Query failed: {query} - {e}")
                results = "[]"
            
            query_time = time.time() - query_start
            latencies.append(query_time)
        
        total_time = time.time() - start_time
        
        # Memory usage
        peak_mem = tracemalloc.get_traced_memory()[1]
        mem_used = (peak_mem - start_mem) / (1024 * 1024)  # MB
        tracemalloc.stop()
        
        # Calculate metrics
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        avg_latency = np.mean(latencies)
        throughput = len(self.queries) / total_time
        
        print(f"Average latency: {avg_latency*1000:.2f}ms")
        print(f"P95 latency: {p95*1000:.2f}ms")
        print(f"P99 latency: {p99*1000:.2f}ms")
        print(f"Throughput: {throughput:.2f} queries/sec")
        print(f"Memory used: {mem_used:.2f} MB")
        
        return {
            'avg_latency': avg_latency,
            'p95_latency': p95,
            'p99_latency': p99,
            'throughput': throughput,
            'memory_mb': mem_used
        }
    
    def run_benchmark(self, index_obj, index_id: str, docs: List):
        """Run complete benchmark for an index."""
        creation_time, disk_size = self.measure_index_creation(index_obj, index_id, docs)
        query_metrics = self.measure_query_performance(index_obj, index_id)
        
        result = {
            'identifier': index_obj.identifier_short,
            'creation_time': creation_time,
            'disk_size_mb': disk_size / (1024*1024),
            **query_metrics
        }
        
        self.results.append(result)
        return result
    
    def save_results(self, output_file: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {output_file}")

def main():
    """Run benchmarks on different index configurations."""
    benchmark = Benchmark()
    
    # Collect documents
    docs = benchmark.collect_documents(max_docs=500)
    
    # Test configurations
    configs = [
        # Elasticsearch baseline
        {
            'class': MyElasticsearchIndex,
            'params': {'core': 'ESIndex', 'info': 'TFIDF', 'dstore': 'DB1', 
                      'qproc': 'TERMatat', 'compr': 'NONE', 'optim': 'Null'},
            'index_id': 'es_baseline'
        },
        # SelfIndex variations for x (index info)
        {
            'class': MySelfIndex,
            'params': {'core': 'SelfIndex', 'info': 'BOOLEAN', 'dstore': 'CUSTOM', 
                      'qproc': 'TERMatat', 'compr': 'NONE', 'optim': 'Null'},
            'index_id': 'self_x1'
        },
        {
            'class': MySelfIndex,
            'params': {'core': 'SelfIndex', 'info': 'WORDCOUNT', 'dstore': 'CUSTOM', 
                      'qproc': 'TERMatat', 'compr': 'NONE', 'optim': 'Null'},
            'index_id': 'self_x2'
        },
        {
            'class': MySelfIndex,
            'params': {'core': 'SelfIndex', 'info': 'TFIDF', 'dstore': 'CUSTOM', 
                      'qproc': 'TERMatat', 'compr': 'NONE', 'optim': 'Null'},
            'index_id': 'self_x3'
        },
        # Compression variations (z)
        {
            'class': MySelfIndex,
            'params': {'core': 'SelfIndex', 'info': 'TFIDF', 'dstore': 'CUSTOM', 
                      'qproc': 'TERMatat', 'compr': 'CODE', 'optim': 'Null'},
            'index_id': 'self_z1'
        },
        {
            'class': MySelfIndex,
            'params': {'core': 'SelfIndex', 'info': 'TFIDF', 'dstore': 'CUSTOM', 
                      'qproc': 'TERMatat', 'compr': 'CLIB', 'optim': 'Null'},
            'index_id': 'self_z2'
        },
        # Query processing variations (q)
        {
            'class': MySelfIndex,
            'params': {'core': 'SelfIndex', 'info': 'TFIDF', 'dstore': 'CUSTOM', 
                      'qproc': 'DOCatat', 'compr': 'NONE', 'optim': 'Null'},
            'index_id': 'self_q_daat'
        },
    ]
    
    # Run benchmarks
    for config in configs:
        try:
            index_obj = config['class'](**config['params'])
            benchmark.run_benchmark(index_obj, config['index_id'], docs)
        except Exception as e:
            print(f"Failed to benchmark {config['index_id']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    benchmark.save_results()
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for result in benchmark.results:
        print(f"\n{result['identifier']}:")
        print(f"  Disk Size: {result['disk_size_mb']:.2f} MB")
        print(f"  P95 Latency: {result['p95_latency']*1000:.2f} ms")
        print(f"  Throughput: {result['throughput']:.2f} q/s")

if __name__ == "__main__":
    main()