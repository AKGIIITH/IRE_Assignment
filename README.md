# IRE Assignment - Information Retrieval and Indexing

This project implements two indexing systems for the Information Retrieval assignment:
- **ESIndex-v1.0**: Elasticsearch-based indexing system
- **SelfIndex-v1.0**: Custom-built indexing system with boolean search and TF-IDF ranking

## Features

### Text Preprocessing
- Tokenization using NLTK
- Stopword removal
- Stemming using Porter Stemmer
- Word frequency analysis with visualization

### ESIndex-v1.0 (Elasticsearch)
- Document indexing with preprocessing
- Full-text search capabilities
- Boolean search support
- Scalable and distributed

### SelfIndex-v1.0 (Custom Implementation)
- Boolean index with document and position IDs
- Inverted index structure
- TF-IDF scoring for ranking
- Boolean query processing (AND, OR, NOT, PHRASE)
- Persistent storage using pickle
- Query parser with proper operator precedence

### Query Processing
- Boolean queries: `information AND retrieval`
- Phrase queries: `"machine learning"`
- Complex queries: `(information AND retrieval) OR "search engine"`
- NOT queries: `search NOT deprecated`

## File Structure

```
IRE_Assignment/
├── main.py                 # Main application demonstrating both systems
├── preprocessor.py         # Text preprocessing module
├── es_indexer.py           # Elasticsearch indexer (ESIndex-v1.0)
├── self_indexer.py         # Custom indexer (SelfIndex-v1.0)
├── query_parser.py         # Boolean query parser
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── index_storage/         # Directory for persistent index files (created at runtime)
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Install and start Elasticsearch for ESIndex-v1.0:
```bash
# Download and start Elasticsearch on localhost:9200
# The system will work without Elasticsearch, using only SelfIndex-v1.0
```

## Usage

Run the main demonstration:
```bash
python main.py
```

This will:
1. Generate word frequency plots
2. Index sample documents using both systems
3. Demonstrate search capabilities
4. Show performance metrics (latency, memory footprint)
5. Display TF-IDF ranking results

## Individual Module Usage

### Text Preprocessing
```python
from preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()
tokens = preprocessor.preprocess_text("Information retrieval systems")
# Output: ['inform', 'retriev', 'system']
```

### Custom Indexer (SelfIndex-v1.0)
```python
from self_indexer import SelfIndexer

indexer = SelfIndexer()
indexer.add_document("doc1", "Title", "Content here")
results = indexer.boolean_search("information AND retrieval")
ranked_results = indexer.search_with_ranking("search query")
```

### Boolean Query Examples
- `information AND retrieval`
- `search OR query`
- `NOT deprecated`
- `"machine learning"`
- `(python OR java) AND NOT deprecated`

## Performance Metrics

The system measures:
- **Latency**: P95 and P99 percentiles for query response time
- **Memory Footprint**: Estimated index size in MB
- **Throughput**: Queries processed per second
- **Functional Metrics**: Search result relevance

## Assignment Requirements Fulfilled

✅ **Data Preprocessing**: Word stemming, stopword removal, punctuation handling  
✅ **Word Frequency Analysis**: Plots with and without preprocessing  
✅ **ESIndex-v1.0**: Elasticsearch integration for indexing and search  
✅ **SelfIndex-v1.0**: Custom indexing implementation  
✅ **Boolean Index**: Document IDs and position IDs  
✅ **TF-IDF Scoring**: Word counts and ranking implementation  
✅ **Query Processing**: Boolean operations with proper precedence  
✅ **Persistence**: Index storage and loading from disk  
✅ **Performance Metrics**: Latency, throughput, and memory measurements  

## Technical Implementation

### Boolean Query Grammar
```
QUERY    := EXPR
EXPR     := TERM | (EXPR) | EXPR AND EXPR | EXPR OR EXPR | NOT EXPR | PHRASE
TERM     := word or "quoted term"
PHRASE   := "quoted phrase"
```

### Operator Precedence (highest to lowest)
1. PHRASE
2. NOT
3. AND
4. OR

### Index Structures
- **Inverted Index**: term → {doc_ids}
- **Positional Index**: term → {doc_id: [positions]}
- **Term Frequencies**: term → {doc_id: frequency}
- **Document Store**: doc_id → {title, content}

## Notes

- The system works without Elasticsearch installed (SelfIndex-v1.0 only)
- All indices are automatically persisted to disk
- NLTK data is automatically downloaded on first run
- Sample documents are included for demonstration
- Word frequency plots are saved as 'word_frequencies.png'
Building an Information Retrieval System from Scratch
