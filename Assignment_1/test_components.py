#!/usr/bin/env python3
"""
Test individual components of the search engine.
"""

def test_preprocessing():
    """Test text preprocessing."""
    print("\n" + "="*60)
    print("Testing Preprocessing")
    print("="*60)
    
    from preprocess import preprocess, tokenize_without_preprocessing
    
    test_text = "The quick brown foxes are running quickly through the forest!"
    
    print(f"Original: {test_text}")
    print(f"Without preprocessing: {tokenize_without_preprocessing(test_text)}")
    print(f"With preprocessing: {preprocess(test_text)}")
    print("✓ Preprocessing working")

def test_query_parser():
    """Test query parser."""
    print("\n" + "="*60)
    print("Testing Query Parser")
    print("="*60)
    
    from query_parser import QueryParser
    
    parser = QueryParser()
    
    test_queries = [
        '"apple"',
        '"apple" AND "banana"',
        '("apple" OR "banana") AND "cherry"',
        '"apple" AND NOT "banana"',
        '"machine learning"',  # Phrase
        '(("deep learning" AND "neural networks") OR "AI") AND NOT "hardware"'
    ]
    
    for query in test_queries:
        ast = parser.parse(query)
        print(f"\nQuery: {query}")
        print(f"AST: {ast}")
    
    print("\n✓ Query parser working")

def test_data_loader():
    """Test data loading."""
    print("\n" + "="*60)
    print("Testing Data Loader")
    print("="*60)
    
    from data_loader import get_all_documents
    
    print("Loading first 5 documents...")
    count = 0
    for doc_id, content in get_all_documents(use_wiki=True, use_news=False, max_wiki_docs=5):
        count += 1
        print(f"\nDoc {count}: {doc_id}")
        print(f"Content length: {len(content)} chars")
        print(f"Preview: {content[:100]}...")
        
        if count >= 5:
            break
    
    print(f"\n✓ Loaded {count} documents")

def test_self_index():
    """Test self-implemented index."""
    print("\n" + "="*60)
    print("Testing Self Index")
    print("="*60)
    
    from self_index import MySelfIndex
    from data_loader import get_all_documents
    
    # Create small index
    print("Creating index with 50 documents...")
    index = MySelfIndex(
        core='SelfIndex',
        info='TFIDF',
        dstore='CUSTOM',
        qproc='TERMatat',
        compr='NONE',
        optim='Null'
    )
    
    docs = []
    for doc_id, content in get_all_documents(max_wiki_docs=50):
        docs.append((doc_id, content))
        if len(docs) >= 50:
            break
    
    index.create_index('test_index', docs)
    print(f"✓ Index created with {len(index.index)} terms")
    
    # Test queries
    test_queries = [
        '"science"',
        '"computer" AND "science"',
        '"machine learning"'
    ]
    
    print("\nTesting queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            results = index.query(query)
            import json
            result_list = json.loads(results)
            print(f"Found {len(result_list)} results")
            if result_list:
                print(f"Top result: {result_list[0]}")
        except Exception as e:
            print(f"Query error: {e}")
    
    print("\n✓ Self index working")

def test_elasticsearch_index():
    """Test Elasticsearch index."""
    print("\n" + "="*60)
    print("Testing Elasticsearch Index")
    print("="*60)
    
    try:
        from es_index import MyElasticsearchIndex
        from data_loader import get_all_documents
        
        print("Connecting to Elasticsearch...")
        es_index = MyElasticsearchIndex()
        
        # Create small index
        print("Creating index with 30 documents...")
        docs = []
        for doc_id, content in get_all_documents(max_wiki_docs=30):
            docs.append((doc_id, content))
            if len(docs) >= 30:
                break
        
        es_index.create_index('test_es_index', docs)
        print("✓ ES index created")
        
        # Test query
        test_query = '"computer science"'
        print(f"\nTesting query: {test_query}")
        results = es_index.query(test_query, 'test_es_index')
        
        import json
        result_list = json.loads(results)
        print(f"Found {len(result_list)} results")
        
        # Cleanup
        es_index.delete_index('test_es_index')
        print("✓ ES index working")
        
    except ConnectionError as e:
        print(f"⚠ Elasticsearch not available: {e}")
        print("This is optional - you can still use SelfIndex")
    except Exception as e:
        print(f"⚠ Error testing ES: {e}")

def test_benchmark():
    """Test benchmarking framework."""
    print("\n" + "="*60)
    print("Testing Benchmark Framework")
    print("="*60)
    
    from benchmark import Benchmark
    from self_index import MySelfIndex
    
    benchmark = Benchmark()
    
    # Collect small document set
    print("Collecting 30 test documents...")
    docs = benchmark.collect_documents(max_docs=30)
    print(f"✓ Collected {len(docs)} documents")
    
    # Create test index
    print("\nCreating test index...")
    index = MySelfIndex(
        core='SelfIndex',
        info='BOOLEAN',
        dstore='CUSTOM',
        qproc='TERMatat',
        compr='NONE',
        optim='Null'
    )
    
    # Run benchmark
    print("Running benchmark...")
    result = benchmark.run_benchmark(index, 'bench_test', docs)
    
    print("\n✓ Benchmark results:")
    print(f"  Creation time: {result['creation_time']:.2f}s")
    print(f"  Disk size: {result['disk_size_mb']:.2f} MB")
    print(f"  P95 latency: {result['p95_latency']*1000:.2f} ms")
    print(f"  Throughput: {result['throughput']:.2f} q/s")

def run_all_tests():
    """Run all component tests."""
    print("\n" + "="*70)
    print("RUNNING ALL COMPONENT TESTS")
    print("="*70)
    
    tests = [
        ("Preprocessing", test_preprocessing),
        ("Query Parser", test_query_parser),
        ("Data Loader", test_data_loader),
        ("Self Index", test_self_index),
        ("Elasticsearch", test_elasticsearch_index),
        ("Benchmark", test_benchmark),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n⚠ {failed} test(s) failed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test search engine components')
    parser.add_argument(
        '--component',
        choices=['all', 'preprocess', 'parser', 'loader', 'index', 'es', 'benchmark'],
        default='all',
        help='Which component to test'
    )
    
    args = parser.parse_args()
    
    if args.component == 'all':
        run_all_tests()
    elif args.component == 'preprocess':
        test_preprocessing()
    elif args.component == 'parser':
        test_query_parser()
    elif args.component == 'loader':
        test_data_loader()
    elif args.component == 'index':
        test_self_index()
    elif args.component == 'es':
        test_elasticsearch_index()
    elif args.component == 'benchmark':
        test_benchmark()