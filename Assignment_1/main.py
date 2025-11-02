#!/usr/bin/env python3
"""
Main script to run the complete search engine pipeline.
"""

import argparse
from pathlib import Path
from data_loader import collect_sample_documents
from preprocess import generate_word_frequency_plots

def setup_environment():
    """Create necessary directories."""
    dirs = ['indices', 'plots', 'data']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    print("Environment setup complete")

def generate_preprocessing_plots():
    """Generate word frequency plots."""
    print("\n" + "="*60)
    print("STEP 1: Generating Word Frequency Plots")
    print("="*60)
    
    docs = collect_sample_documents(n=100)
    generate_word_frequency_plots(docs, output_prefix='word_freq')
    print("Word frequency plots generated successfully")

def run_benchmarks():
    """Run all benchmarks."""
    print("\n" + "="*60)
    print("STEP 2: Running Benchmarks")
    print("="*60)
    
    from benchmark import main as benchmark_main
    benchmark_main()

def generate_plots():
    """Generate all comparison plots."""
    print("\n" + "="*60)
    print("STEP 3: Generating Comparison Plots")
    print("="*60)
    
    from plot_generator import PlotGenerator
    plotter = PlotGenerator()
    plotter.generate_all_plots()

def create_sample_queries():
    """Create sample query file if it doesn't exist."""
    query_file = Path("diverse-queries.json")
    
    if not query_file.exists():
        import json
        sample_queries = [
            '"machine learning"',
            '"artificial intelligence"',
            '"data science"',
            '"python programming"',
            '"neural networks"',
            '"deep learning"',
            '"natural language processing"',
            '"computer vision"',
            '"algorithm" AND "complexity"',
            '"database" OR "storage"',
            '"web" AND "development"',
            '"software" AND NOT "hardware"',
            '"cloud computing"',
            '"distributed systems"',
            '"information retrieval"',
        ]
        
        with open(query_file, 'w') as f:
            json.dump(sample_queries, f, indent=2)
        
        print(f"Created sample query file: {query_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Search Engine Implementation and Benchmarking'
    )
    parser.add_argument(
        '--step',
        choices=['all', 'preprocess', 'benchmark', 'plots'],
        default='all',
        help='Which step to run (default: all)'
    )
    parser.add_argument(
        '--skip-es',
        action='store_true',
        help='Skip Elasticsearch benchmarks'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_environment()
    create_sample_queries()
    
    # Run requested steps
    if args.step in ['all', 'preprocess']:
        try:
            generate_preprocessing_plots()
        except Exception as e:
            print(f"Error in preprocessing: {e}")
    
    if args.step in ['all', 'benchmark']:
        try:
            run_benchmarks()
        except Exception as e:
            print(f"Error in benchmarking: {e}")
            import traceback
            traceback.print_exc()
    
    if args.step in ['all', 'plots']:
        try:
            generate_plots()
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - word_freq_before.png, word_freq_after.png")
    print("  - benchmark_results.json")
    print("  - plots/*.png")
    print("\nIndex files stored in: indices/")

if __name__ == "__main__":
    main()