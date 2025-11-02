# Modular Search Engine Implementation

A comprehensive search engine implementation featuring both Elasticsearch baseline and a modular from-scratch indexing system with configurable components.

## Features

- **Data Loading**: Wikipedia and News datasets
- **Text Preprocessing**: Tokenization, stemming, stop word removal
- **Elasticsearch Baseline**: Full-featured ES index for comparison
- **Modular Self-Index**: Configurable indexing with multiple strategies:
  - Index types: Boolean, Word Count, TF-IDF
  - Datastores: Custom (pickle), RocksDB, PostgreSQL
  - Compression: None, Variable-Byte, Zlib
  - Query processing: Term-at-a-Time (TAAT), Document-at-a-Time (DAAT)
  - Optimizations: Skip pointers
- **Boolean Query Parser**: Supports AND, OR, NOT, and PHRASE queries
- **Comprehensive Benchmarking**: Latency (P95, P99), throughput, memory footprint
- **Visualization**: Automatic plot generation for all metrics

## Project Structure

```
.
├── index_base.py          # Base class with strategy enums
├── preprocess.py          # Text preprocessing utilities
├── data_loader.py         # Dataset loading functions
├── query_parser.py        # Boolean query parser
├── query_engine.py        # Query execution engine
├── self_index.py          # Modular self-implemented index
├── es_index.py            # Elasticsearch wrapper
├── benchmark.py           # Benchmarking framework
├── plot_generator.py      # Plot generation
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data

The code will automatically download required NLTK data on first run, but you can pre-download:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 3. (Optional) Install Elasticsearch

For baseline comparison, install and run Elasticsearch:

```bash
# Download from https://www.elastic.co/downloads/elasticsearch
# Or using Docker:
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.9.0
```

### 4. (Optional) Install Database Backends

For RocksDB support:
```bash
pip install rocksdb-python
```

For PostgreSQL support:
```bash
pip install psycopg2-binary
```

## Quick Start

### Run Complete Pipeline

```bash
python main.py
```

This will:
1. Generate word frequency plots (before/after preprocessing)
2. Run benchmarks on all index configurations
3. Generate comparison plots

### Run Individual Steps

```bash
# Only preprocessing
python main.py --step preprocess

# Only benchmarks
python main.py --step benchmark

# Only plot generation
python main.py --step plots
```

## Usage Examples

### Creating an Index

```python
from self_index import MySelfIndex
from data_loader import get_all_documents

# Create a TF-IDF index with custom storage
index = MySelfIndex(
    core='SelfIndex',
    info='TFIDF',
    dstore='CUSTOM',
    qproc='TERMatat',
    compr='NONE',
    optim='Null'
)

# Load documents
docs = list(get_all_documents(max_wiki_docs=1000))

# Create index
index.create_index('my_index', docs)
```

### Querying

```python
# Simple term query
results = index.query('"machine learning"')

# Boolean query
results = index.query('("python" AND "programming") OR "java"')

# Phrase query
results = index.query('"natural language processing"')

# Complex query
results = index.query('("AI" OR "ML") AND NOT "hardware"')
```

### Custom Benchmarking

```python
from benchmark import Benchmark

benchmark = Benchmark(query_file='my_queries.json')
docs = benchmark.collect_documents(max_docs=500)

# Test specific configuration
index = MySelfIndex(
    core='SelfIndex',
    info='TFIDF',
    dstore='CUSTOM',
    qproc='TERMatat',
    compr='CLIB',
    optim='Skipping'
)

results = benchmark.run_benchmark(index, 'test_index', docs)
print(f"P95 Latency: {results['p95_latency']*1000:.2f}ms")
print(f"Throughput: {results['throughput']:.2f} q/s")
```

## Index Configuration Options

### Index Information (x parameter)
- **BOOLEAN** (x=1): Document IDs and positions only
- **WORDCOUNT** (x=2): Adds term frequency counts
- **TFIDF** (x=3): Full TF-IDF scoring

### Datastore (y parameter)
- **CUSTOM** (y=1): Pickle serialization to disk
- **DB1** (y=2): RocksDB key-value store
- **DB2** (y=2): PostgreSQL with GIN indexing

### Compression (z parameter)
- **NONE** (z=0): No compression
- **CODE** (z=1): Variable-byte encoding
- **CLIB** (z=2): Zlib compression

### Query Processing (q parameter)
- **TERMatat** (q=T): Term-at-a-Time processing
- **DOCatat** (q=D): Document-at-a-Time processing

### Optimizations (i parameter)
- **Null** (i=0): No optimizations
- **Skipping** (i=1): Skip pointers for AND operations

## Query Syntax

The system supports boolean queries with the following operators:

### Operators (by precedence)
1. **PHRASE**: `"term1 term2"` - Matches exact phrase
2. **NOT**: `NOT "term"` - Excludes documents
3. **AND**: `"term1" AND "term2"` - Both terms required
4. **OR**: `"term1" OR "term2"` - Either term matches

### Examples

```python
# Single term
'"machine"'

# Phrase query
'"machine learning"'

# Boolean AND
'"python" AND "programming"'

# Boolean OR
'"java" OR "javascript"'

# NOT operator
'"software" AND NOT "hardware"'

# Complex with parentheses
'("AI" OR "ML") AND ("python" OR "java")'

# Multiple operations
'("deep learning" AND "neural networks") OR "computer vision"'
```

## Benchmark Metrics

### Metric A: Latency
- Average, P95, and P99 query response times
- Measured in milliseconds

### Metric B: Throughput
- Queries per second (QPS)
- Total queries / total execution time

### Metric C: Memory Footprint
- Disk size: Index storage on disk (MB)
- RAM usage: Peak memory during query execution (MB)

### Metric D: Functional Metrics
- Precision and recall (if ground truth available)
- Result relevance scoring

## Generated Outputs

### Word Frequency Plots
- `word_freq_before.png`: Top words before preprocessing
- `word_freq_after.png`: Top words after stemming/stop word removal

### Benchmark Results
- `benchmark_results.json`: Raw benchmark data

### Comparison Plots
- `plot_c_index_types.png`: Disk size vs index type
- `plot_a_datastores.png`: Latency vs datastore
- `plot_ab_compression.png`: Latency & throughput vs compression
- `plot_a_optimization.png`: Latency vs skip pointers
- `plot_ac_query_processing.png`: Latency & memory vs query processing
- `comparison_summary.png`: Overall comparison dashboard

## Dataset Information

### Wikipedia
- Source: Hugging Face `wikimedia/wikipedia` dataset
- Split: `20231101.en`
- Streaming: Yes (doesn't download entire dataset)
- Fields: `id`, `title`, `text`

### News Dataset
- Source: Local directory `News_Dataset/`
- Format: JSON files organized in topic folders
- Fields: `title`, `text`

## Performance Tips

1. **For faster indexing**: Use fewer documents initially
   ```python
   docs = list(get_all_documents(max_wiki_docs=100))
   ```

2. **For lower memory usage**: Use compression
   ```python
   compr='CLIB'  # Use zlib compression
   ```

3. **For faster queries**: Use skip pointers
   ```python
   optim='Skipping'
   ```

4. **For best ranking**: Use TF-IDF
   ```python
   info='TFIDF'
   ```

## Troubleshooting

### Elasticsearch Connection Error
- Ensure Elasticsearch is running on `localhost:9200`
- Check with: `curl http://localhost:9200`
- Or skip ES tests: `python main.py --skip-es`

### RocksDB Import Error
- RocksDB is optional
- System falls back to pickle if unavailable
- Install: `pip install rocksdb-python`

### Out of Memory
- Reduce `max_wiki_docs` in benchmarks
- Use compression: `compr='CLIB'`
- Process in batches

### Slow Query Performance
- Build index with skip pointers: `optim='Skipping'`
- Use DAAT instead of TAAT: `qproc='DOCatat'`
- Enable compression to reduce I/O: `compr='CLIB'`

## Contributing

When adding new features:

1. Update `IndexBase` enums in `index_base.py`
2. Implement strategy in `MySelfIndex`
3. Add tests to `benchmark.py`
4. Update plot generation in `plot_generator.py`

## License

This is an educational project for information retrieval coursework.

## Authors

Ayush Kumar Gupta
Course: Information Retrieval Systems