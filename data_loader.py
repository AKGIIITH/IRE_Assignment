import json
import os
from typing import Generator, Tuple
from datasets import load_dataset

def load_wikipedia(max_docs: int = None) -> Generator[Tuple[str, str], None, None]:
    """
    Load Wikipedia dataset from Hugging Face.
    
    Args:
        max_docs: Maximum number of documents to load (None for all)
        
    Yields:
        Tuples of (doc_id, content) where content is title + text
    """
    print("Loading Wikipedia dataset...")
    
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        streaming=True,
        split="train",
        trust_remote_code=True
    )
    
    count = 0
    for doc in dataset:
        doc_id = f"wiki_{doc['id']}"
        title = doc.get('title', '')
        text = doc.get('text', '')
        content = f"{title} {text}".strip()
        
        if content:
            yield (doc_id, content)
            count += 1
            
            if max_docs and count >= max_docs:
                break
    
    print(f"Loaded {count} Wikipedia documents")

def load_news_dataset(news_dir: str = "News_Dataset") -> Generator[Tuple[str, str], None, None]:
    """
    Load news dataset from local directory structure.
    
    Args:
        news_dir: Root directory containing news articles
        
    Yields:
        Tuples of (doc_id, content) where content is title + text
    """
    if not os.path.exists(news_dir):
        print(f"Warning: {news_dir} not found. Skipping news dataset.")
        return
    
    print(f"Loading news dataset from {news_dir}...")
    
    count = 0
    for root, dirs, files in os.walk(news_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                    
                    # Generate unique doc_id
                    doc_id = f"news_{count}"
                    
                    # Extract title and text
                    title = article.get('title', '')
                    text = article.get('text', '')
                    content = f"{title} {text}".strip()
                    
                    if content:
                        yield (doc_id, content)
                        count += 1
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
    
    print(f"Loaded {count} news documents")

def get_all_documents(use_wiki: bool = True, use_news: bool = True, 
                      max_wiki_docs: int = 1000, news_dir: str = "News_Dataset") -> Generator[Tuple[str, str], None, None]:
    """
    Load all documents from available sources.
    
    Args:
        use_wiki: Whether to load Wikipedia data
        use_news: Whether to load news data
        max_wiki_docs: Maximum Wikipedia documents to load
        news_dir: Directory containing news articles
        
    Yields:
        Tuples of (doc_id, content)
    """
    if use_wiki:
        yield from load_wikipedia(max_docs=max_wiki_docs)
    
    if use_news:
        yield from load_news_dataset(news_dir=news_dir)

# Helper function to collect documents for preprocessing
def collect_sample_documents(n: int = 100) -> list[str]:
    """Collect sample documents for word frequency analysis."""
    documents = []
    for doc_id, content in get_all_documents(max_wiki_docs=n):
        documents.append(content)
        if len(documents) >= n:
            break
    return documents