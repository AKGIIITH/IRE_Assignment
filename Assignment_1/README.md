# Modular Search Engine Implementation

A comprehensive search engine implementation featuring both an Elasticsearch baseline and a modular from-scratch indexing system with configurable components. This project is designed to explore and evaluate the internal mechanics of a search engine, from text preprocessing and indexing to query execution and performance benchmarking.

## Features

- **Data Loading**: Supports streaming from the Wikipedia dataset on Hugging Face and loading from a local News dataset.
- **Text Preprocessing**: Includes tokenization, Porter stemming, and stop-word removal using NLTK.
- **Elasticsearch Baseline**: A full-featured Elasticsearch index is used for performance comparison.
- **Modular Self-Index**: A highly configurable custom index implementation supports multiple strategies:
  - **Index Information**: Boolean, Word Count, and TF-IDF scoring.
  - **Datastores**: Custom (pickle), RocksDB, and a placeholder for PostgreSQL.
  - **Compression**: No compression, a custom Variable-Byte encoding, and Zlib.
  - **Query Processing**: Term-at-a-Time (TAAT) and Document-at-a-Time (DAAT).
  - **Optimizations**: Skip pointers to accelerate query processing.
- **Boolean Query Parser**: A robust parser based on the Shunting-yard algorithm that supports `AND`, `OR`, `NOT`, and `PHRASE` queries with correct operator precedence.
- **Comprehensive Benchmarking**: Measures P95/P99 latency, throughput (QPS), index creation time, and memory footprint (disk and RAM).
- **Visualization**: Automatically generates plots for all benchmark comparisons.

## Project Structure

The project is organized into several key modules, each responsible for a specific part of the search engine pipeline.

```
.
├── main.py                # Main execution script to run the pipeline
├── benchmark.py           # Benchmarking framework to run experiments
├── plot_generator.py      # Generates plots from benchmark results
├── requirements.txt       # Python dependencies
├── README.md              # This file
|
├── index_base.py          # Abstract base class for indices and configuration enums
├── self_index.py          # Modular self-implemented index
├── es_index.py            # Elasticsearch index implementation
|
├── query_parser.py        # Boolean query parser (Shunting-yard)
├── query_engine.py        # Executes parsed query ASTs
|
├── preprocess.py          # Text preprocessing utilities (stemming, stopwords)
├── data_loader.py         # Loads data from Wikipedia and local news files
|
├── test_components.py     # Unit tests for individual modules
├── diverse-queries.json   # A diverse set of queries for benchmarking
└── docker-compose.yml     # Docker configuration for Elasticsearch
```

## Setup and Installation

Follow these steps to set up the environment and run the project.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Assignment_1
```

### 2. Install Dependencies

Install the required Python packages using pip. It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The script will automatically download necessary NLTK data (`punkt`, `stopwords`) on the first run.

### 3. Start Elasticsearch (Optional)

An Elasticsearch instance is required to run the baseline benchmark. The easiest way to start one is with Docker Compose.

```bash
sudo docker compose up -d
```

This will start an Elasticsearch container running on `localhost:9200`. You can verify it's running with `curl http://localhost:9200`.

### 4. Install Optional Database Backends (Optional)

The self-implemented index supports different datastores. To test RocksDB, you need to install its Python binding.

```bash
# For RocksDB support (y=2)
pip install rocksdb-python
```

PostgreSQL support is not fully implemented. If it were, you would install `psycopg2-binary`.

## How to Run

The `main.py` script is the primary entry point for running the entire pipeline, from preprocessing to benchmarking and plot generation.

### Run the Complete Pipeline

To run all steps sequentially, execute:

```bash
python main.py
```

This command will:
1.  **Generate Preprocessing Plots**: Create `word_freq_before.png` and `word_freq_after.png`.
2.  **Run Benchmarks**: Execute all defined index configurations, creating and querying each one. The raw results are saved to `benchmark_results.json`.
3.  **Generate Result Plots**: Create all comparison plots in the `plots/` directory from the benchmark results.

If you do not have Docker or do not wish to run the Elasticsearch benchmark, use the `--skip-es` flag:
```bash
python main.py --skip-es
```

### Run Individual Steps

You can also run each major step of the pipeline independently.

1.  **Generate Preprocessing Plots**:
    ```bash
    python main.py --step preprocess
    ```
    *   **Output**: `word_freq_before.png`, `word_freq_after.png`.

2.  **Run Benchmarks**:
    ```bash
    python main.py --step benchmark
    ```
    *   **Output**: `benchmark_results.json`, and index files are stored in the `indices/` directory.

3.  **Generate Plots from Existing Results**:
    ```bash
    python main.py --step plots
    ```
    *   **Output**: All plots are generated in the `plots/` directory from `benchmark_results.json`.

### Running Tests

To verify that all components are working correctly, you can run the test suite.

```bash
python test_components.py
```

You can also test a specific component:
```bash
python test_components.py --component parser
```

## Generated Outputs

After running the complete pipeline, the following files and directories will be generated:

-   `word_freq_before.png` / `word_freq_after.png`: Word frequency distribution plots.
-   `benchmark_results.json`: A JSON file containing the raw performance metrics for each benchmarked configuration.
-   `indices/`: A directory containing the persisted index files for each configuration of `MySelfIndex`.
-   `plots/`: A directory containing all generated plots for performance comparison:
    -   `plot_c_index_types.png`: Disk size vs. index type.
    -   `plot_a_datastores.png`: Latency vs. datastore.
    -   `plot_ab_compression.png`: Latency & throughput vs. compression.
    -   `plot_a_optimization.png`: Latency vs. skip pointers.
    -   `plot_ac_query_processing.png`: Latency & memory vs. query processing mode.
    -   `plot_comparison_summary.png`: An overall dashboard comparing key metrics across configurations.

## Index Configuration Options

The `MySelfIndex` class is modular and can be configured with different strategies for indexing, storage, and querying.

-   **Index Information (x)**:
    -   `BOOLEAN` (x=1): Stores document IDs and term positions.
    -   `WORDCOUNT` (x=2): Adds term frequency.
    -   `TFIDF` (x=3): Stores full TF-IDF scores for ranking.
-   **Datastore (y)**:
    -   `CUSTOM` (y=1): Uses Python's `pickle` for serialization to disk.
    -   `DB1` (y=2): Uses RocksDB as a key-value store.
-   **Compression (z)**:
    -   `NONE` (z=0): No compression.
    -   `CODE` (z=1): Variable-byte encoding (placeholder).
    -   `CLIB` (z=2): Zlib compression.
-   **Query Processing (q)**:
    -   `TERMatat` (q=T): Term-at-a-Time processing.
    -   `DOCatat` (q=D): Document-at-a-Time processing.
-   **Optimizations (i)**:
    -   `Null` (i=0): No optimizations.
    -   `Skipping` (i=1): Uses skip pointers to speed up `AND` operations.

## Query Syntax

The system supports boolean queries with `AND`, `OR`, `NOT`, and phrase searches.

-   **Precedence**: `PHRASE` > `NOT` > `AND` > `OR`.
-   **Terms**: Single words must be enclosed in double quotes (e.g., `"apple"`).
-   **Phrases**: Multi-word phrases are also enclosed in double quotes (e.g., `"machine learning"`).

**Examples**:
- `("AI" OR "ML") AND ("python" OR "java")`
- `"software" AND NOT "hardware"`
- `("deep learning" AND "neural networks")