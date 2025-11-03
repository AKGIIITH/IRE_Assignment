"""
Quick test script to verify all components work correctly.
"""

def test_preprocessing():
    """Test preprocessing functionality."""
    print("Testing preprocessing...")
    from preprocess import preprocess
    
    text = "The quick brown foxes jumped over the lazy dogs. Natural Language Processing is amazing!"
    tokens = preprocess(text)
    
    print(f"  Input: {text}")
    print(f"  Tokens: {tokens}")
    assert len(tokens) > 0, "Preprocessing should return tokens"
    print("  Preprocessing works\n")

def test_query_parser():
    """Test query parser."""
    print("Testing query parser...")
    from query_parser import QueryParser, TermNode, AndNode, OrNode, NotNode, PhraseNode
    
    parser = QueryParser()
    
    # Test simple term
    ast = parser.parse('"machine"')
    assert isinstance(ast, TermNode), "Should parse as TermNode"
    print(f"  Query: \"machine\" -> {ast}")
    
    # Test phrase
    ast = parser.parse('"machine learning"')
    assert isinstance(ast, PhraseNode), "Should parse as PhraseNode"
    print(f"  Query: \"machine learning\" -> {ast}")
    
    # Test AND
    ast = parser.parse('"python" AND "programming"')
    assert isinstance(ast, AndNode), "Should parse as AndNode"
    print(f"  Query: \"python\" AND \"programming\" -> {ast}")
    
    # Test complex
    ast = parser.parse('("AI" OR "ML") AND "python"')
    assert isinstance(ast, AndNode), "Should parse as AndNode"
    print(f"  Query: (\"AI\" OR \"ML\") AND \"python\" -> {ast}")
    
    print("  Query parser works\n")

def test_index_creation():
    """Test index creation and querying."""
    print("Testing index creation...")
    from self_index import MySelfIndex
    
    # Create a simple index
    index = MySelfIndex(
        core='SelfIndex',
        info='TFIDF',
        dstore='CUSTOM',
        qproc='TERMatat',
        compr='NONE',
        optim='Null'
    )
    
    # Sample documents
    docs = [
        ('doc1', 'Python is a programming language. Python is popular.'),
        ('doc2', 'Machine learning uses Python and algorithms.'),
        ('doc3', 'Java is also a programming language.'),
    ]
    
    print(f"  Creating index with {len(docs)} documents...")
    index.create_index('test_index', docs)
    print(f"  Index identifier: {index.identifier_short}")
    
    # Test query
    print("  Testing query: \"python\"")
    results = index.query('"python"')
    print(f"  Results: {results}")
    
    print("  Index creation and querying works\n")

def test_data_loader():
    """Test data loading."""
    print("Testing data loader...")
    from data_loader import get_all_documents
    
    docs = []
    for doc_id, content in get_all_documents(use_wiki=True, use_news=False, max_wiki_docs=5):
        docs.append((doc_id, content[:100]))  # First 100 chars
        if len(docs) >= 3:
            break
    
    print(f"  Loaded {len(docs)} sample documents:")
    for doc_id, content in docs:
        print(f"    - {doc_id}: {content}...")
    
    assert len(docs) > 0, "Should load at least some documents"
    print("  Data loader works\n")

def main():
    """Run all tests."""
    print("="*60)
    print("QUICK COMPONENT TEST")
    print("="*60)
    print()
    
    try:
        test_preprocessing()
        test_query_parser()
        test_data_loader()
        test_index_creation()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print()
        print("The system is ready to run!")
        print("Execute: python main.py")
        
        # Clean up before exit to prevent PyGILState errors
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    import os
    
    # Set environment variable to suppress PyArrow threading warnings
    os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'
    
    exit_code = main()
    
    # Force immediate exit to avoid cleanup issues
    os._exit(exit_code)