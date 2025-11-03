"""Check if plots are consistent with benchmark.json and regenerate if needed."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_benchmark_data():
    """Load benchmark results."""
    with open('benchmark_results.json', 'r') as f:
        return json.load(f)

def check_and_plot():
    """Check plots against benchmark data and regenerate if needed."""
    
    print("Loading benchmark data...")
    data = load_benchmark_data()
    
    print(f"Found {len(data)} benchmark configurations\n")
    
    # Organize data by categories
    by_datastore = {}
    by_compression = {}
    by_index_type = {}
    by_query_proc = {}
    by_optimization = {}
    
    for config in data:
        name = config['identifier']  # Changed from 'name' to 'identifier'
        
        # Extract configuration from name (format: SelfIndex_i{x}d{y}c{z}q{w}o{opt})
        parts = name.split('_')
        if len(parts) < 2:
            continue
            
        config_str = parts[1]  # e.g., "i3d2c1qTo0"
        
        # Parse configuration
        idx_type = config_str[1] if len(config_str) > 1 else '?'
        datastore = config_str[3] if len(config_str) > 3 else '?'
        compression = config_str[5] if len(config_str) > 5 else '?'
        query_proc = config_str[7] if len(config_str) > 7 else '?'
        optimization = config_str[9:] if len(config_str) > 9 else '0'
        
        # Group by datastore (y)
        datastore_name = {'1': 'Custom/Pickle', '2': 'SQLite', '3': 'PostgreSQL'}.get(datastore, f'DS_{datastore}')
        if datastore_name not in by_datastore:
            by_datastore[datastore_name] = []
        by_datastore[datastore_name].append(config)
        
        # Group by compression (z)
        comp_name = {'1': 'None', '2': 'VByte', '3': 'zlib'}.get(compression, f'Comp_{compression}')
        if comp_name not in by_compression:
            by_compression[comp_name] = []
        by_compression[comp_name].append(config)
        
        # Group by index type (x)
        idx_name = {'1': 'Boolean', '2': 'WordCount', '3': 'TF-IDF'}.get(idx_type, f'Idx_{idx_type}')
        if idx_name not in by_index_type:
            by_index_type[idx_name] = []
        by_index_type[idx_name].append(config)
        
        # Group by query processing (q)
        qproc_name = {'T': 'TAAT', 'D': 'DAAT'}.get(query_proc, f'QP_{query_proc}')
        if qproc_name not in by_query_proc:
            by_query_proc[qproc_name] = []
        by_query_proc[qproc_name].append(config)
        
        # Group by optimization (o)
        opt_name = {'0': 'None', 'sp': 'Skip Pointers', 'th': 'Threshold', 'es': 'Early Stop'}.get(optimization, optimization)
        if opt_name not in by_optimization:
            by_optimization[opt_name] = []
        by_optimization[opt_name].append(config)
    
    # Print summary
    print("Configuration breakdown:")
    print(f"  Datastores: {list(by_datastore.keys())}")
    print(f"  Compressions: {list(by_compression.keys())}")
    print(f"  Index types: {list(by_index_type.keys())}")
    print(f"  Query processing: {list(by_query_proc.keys())}")
    print(f"  Optimizations: {list(by_optimization.keys())}")
    print()
    
    # Calculate averages
    def calc_avg(configs_list, key):
        if not configs_list:
            return 0
        values = [c[key] for c in configs_list if key in c]
        return sum(values) / len(values) if values else 0
    
    # Generate plots
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Datastore comparison (y)
    print("Generating datastore comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Datastore Comparison (y parameter)', fontsize=16, fontweight='bold')
    
    datastores = list(by_datastore.keys())
    
    # Index time
    ax = axes[0, 0]
    index_times = [calc_avg(by_datastore[ds], 'creation_time') for ds in datastores]
    ax.bar(datastores, index_times, color='steelblue')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Index Creation Time')
    ax.grid(axis='y', alpha=0.3)
    
    # Query time
    ax = axes[0, 1]
    query_times = [calc_avg(by_datastore[ds], 'avg_latency') for ds in datastores]
    ax.bar(datastores, query_times, color='coral')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Average Query Latency')
    ax.grid(axis='y', alpha=0.3)
    
    # Index size
    ax = axes[1, 0]
    index_sizes = [calc_avg(by_datastore[ds], 'disk_size_mb') for ds in datastores]
    ax.bar(datastores, index_sizes, color='seagreen')
    ax.set_ylabel('Size (MB)')
    ax.set_title('Index Size')
    ax.grid(axis='y', alpha=0.3)
    
    # Throughput
    ax = axes[1, 1]
    throughputs = [calc_avg(by_datastore[ds], 'throughput') for ds in datastores]
    ax.bar(datastores, throughputs, color='mediumpurple')
    ax.set_ylabel('Queries/sec')
    ax.set_title('Query Throughput')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'plot_a_datastores.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: plot_a_datastores.png")
    plt.close()
    
    # 2. Compression comparison (z)
    print("Generating compression comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Compression Strategy Comparison (z parameter)', fontsize=16, fontweight='bold')
    
    compressions = list(by_compression.keys())
    
    # Index time
    ax = axes[0, 0]
    index_times = [calc_avg(by_compression[c], 'creation_time') for c in compressions]
    ax.bar(compressions, index_times, color='steelblue')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Index Creation Time')
    ax.grid(axis='y', alpha=0.3)
    
    # Query time
    ax = axes[0, 1]
    query_times = [calc_avg(by_compression[c], 'avg_latency') for c in compressions]
    ax.bar(compressions, query_times, color='coral')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Average Query Latency')
    ax.grid(axis='y', alpha=0.3)
    
    # Index size
    ax = axes[1, 0]
    index_sizes = [calc_avg(by_compression[c], 'disk_size_mb') for c in compressions]
    ax.bar(compressions, index_sizes, color='seagreen')
    ax.set_ylabel('Size (MB)')
    ax.set_title('Index Size')
    ax.grid(axis='y', alpha=0.3)
    
    # Compression ratio
    ax = axes[1, 1]
    if compressions:
        baseline_size = calc_avg(by_compression[compressions[0]], 'disk_size_mb')
        ratios = [baseline_size / calc_avg(by_compression[c], 'disk_size_mb') if calc_avg(by_compression[c], 'disk_size_mb') > 0 else 0 for c in compressions]
        ax.bar(compressions, ratios, color='mediumpurple')
        ax.set_ylabel('Compression Ratio')
        ax.set_title('Compression Ratio (vs baseline)')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'plot_ab_compression.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: plot_ab_compression.png")
    plt.close()
    
    # 3. Index type comparison (x)
    print("Generating index type comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Index Type Comparison (x parameter)', fontsize=16, fontweight='bold')
    
    index_types = list(by_index_type.keys())
    
    # Index time
    ax = axes[0, 0]
    index_times = [calc_avg(by_index_type[it], 'creation_time') for it in index_types]
    ax.bar(index_types, index_times, color='steelblue')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Index Creation Time')
    ax.grid(axis='y', alpha=0.3)
    
    # Query time
    ax = axes[0, 1]
    query_times = [calc_avg(by_index_type[it], 'avg_latency') for it in index_types]
    ax.bar(index_types, query_times, color='coral')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Average Query Latency')
    ax.grid(axis='y', alpha=0.3)
    
    # Throughput
    ax = axes[1, 0]
    throughputs = [calc_avg(by_index_type[it], 'throughput') for it in index_types]
    ax.bar(index_types, throughputs, color='seagreen')
    ax.set_ylabel('Queries/sec')
    ax.set_title('Query Throughput')
    ax.grid(axis='y', alpha=0.3)
    
    # Memory usage
    ax = axes[1, 1]
    memory = [calc_avg(by_index_type[it], 'memory_mb') for it in index_types]
    ax.bar(index_types, memory, color='mediumpurple')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Usage')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'plot_c_index_types.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: plot_c_index_types.png")
    plt.close()
    
    # 4. Query processing comparison (q)
    print("Generating query processing comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Query Processing Strategy Comparison (q parameter)', fontsize=16, fontweight='bold')
    
    query_procs = list(by_query_proc.keys())
    
    if len(query_procs) > 0:
        # Query time
        ax = axes[0, 0]
        query_times = [calc_avg(by_query_proc[qp], 'avg_latency') for qp in query_procs]
        ax.bar(query_procs, query_times, color='coral')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Average Query Latency')
        ax.grid(axis='y', alpha=0.3)
        
        # P95 latency
        ax = axes[0, 1]
        p95_latencies = [calc_avg(by_query_proc[qp], 'p95_latency') for qp in query_procs]
        ax.bar(query_procs, p95_latencies, color='seagreen')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('P95 Latency')
        ax.grid(axis='y', alpha=0.3)
        
        # P99 latency
        ax = axes[1, 0]
        p99_latencies = [calc_avg(by_query_proc[qp], 'p99_latency') for qp in query_procs]
        ax.bar(query_procs, p99_latencies, color='mediumpurple')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('P99 Latency')
        ax.grid(axis='y', alpha=0.3)
        
        # Throughput
        ax = axes[1, 1]
        throughputs = [calc_avg(by_query_proc[qp], 'throughput') for qp in query_procs]
        ax.bar(query_procs, throughputs, color='steelblue')
        ax.set_ylabel('Queries/sec')
        ax.set_title('Query Throughput')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'plot_ac_query_processing.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: plot_ac_query_processing.png")
    plt.close()
    
    # 5. Optimization comparison (o)
    print("Generating optimization comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimization Strategy Comparison (o parameter)', fontsize=16, fontweight='bold')
    
    optimizations = list(by_optimization.keys())
    
    if len(optimizations) > 0:
        # Query time
        ax = axes[0, 0]
        query_times = [calc_avg(by_optimization[opt], 'avg_latency') for opt in optimizations]
        ax.bar(optimizations, query_times, color='coral')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Average Query Latency')
        ax.grid(axis='y', alpha=0.3)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
        
        # Throughput
        ax = axes[0, 1]
        throughputs = [calc_avg(by_optimization[opt], 'throughput') for opt in optimizations]
        ax.bar(optimizations, throughputs, color='seagreen')
        ax.set_ylabel('Queries/sec')
        ax.set_title('Query Throughput')
        ax.grid(axis='y', alpha=0.3)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
        
        # Index size
        ax = axes[1, 0]
        index_sizes = [calc_avg(by_optimization[opt], 'disk_size_mb') for opt in optimizations]
        ax.bar(optimizations, index_sizes, color='mediumpurple')
        ax.set_ylabel('Size (MB)')
        ax.set_title('Index Size')
        ax.grid(axis='y', alpha=0.3)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
        
        # Speedup
        ax = axes[1, 1]
        if optimizations:
            baseline_time = calc_avg(by_optimization[optimizations[0]], 'avg_latency')
            speedups = [baseline_time / calc_avg(by_optimization[opt], 'avg_latency') if calc_avg(by_optimization[opt], 'avg_latency') > 0 else 0 for opt in optimizations]
            ax.bar(optimizations, speedups, color='steelblue')
            ax.set_ylabel('Speedup Factor')
            ax.set_title('Query Speedup (vs baseline)')
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
            ax.grid(axis='y', alpha=0.3)
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'plot_a_optimization.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: plot_a_optimization.png")
    plt.close()
    
    # 6. Overall comparison summary
    print("Generating comparison summary plot...")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Benchmark Results Summary', fontsize=18, fontweight='bold')
    
    # Top metrics by configuration
    all_configs = data.copy()
    all_configs.sort(key=lambda x: x['avg_latency'])
    
    # Best query time
    ax = fig.add_subplot(gs[0, :])
    top_n = min(10, len(all_configs))
    names = [c['identifier'].replace('SelfIndex_', '').replace('ESIndex_', 'ES_') for c in all_configs[:top_n]]
    times = [c['avg_latency'] for c in all_configs[:top_n]]
    ax.barh(names, times, color='coral')
    ax.set_xlabel('Query Latency (seconds)')
    ax.set_title(f'Top {top_n} Configurations by Query Speed', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Index size distribution
    ax = fig.add_subplot(gs[1, 0])
    sizes = [c['disk_size_mb'] for c in all_configs]
    ax.hist(sizes, bins=15, color='seagreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Index Size (MB)')
    ax.set_ylabel('Frequency')
    ax.set_title('Index Size Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    # Query time distribution
    ax = fig.add_subplot(gs[1, 1])
    times = [c['avg_latency'] for c in all_configs]
    ax.hist(times, bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Query Latency (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Query Latency Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    # Throughput distribution
    ax = fig.add_subplot(gs[1, 2])
    throughputs = [c['throughput'] for c in all_configs]
    ax.hist(throughputs, bins=15, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Throughput (queries/sec)')
    ax.set_ylabel('Frequency')
    ax.set_title('Throughput Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    # Trade-off: Query time vs Index size
    ax = fig.add_subplot(gs[2, 0])
    times = [c['avg_latency'] for c in all_configs]
    sizes = [c['disk_size_mb'] for c in all_configs]
    colors = ['red' if 'ES' in c['identifier'] else 'steelblue' for c in all_configs]
    ax.scatter(times, sizes, alpha=0.6, s=50, c=colors)
    ax.set_xlabel('Query Latency (seconds)')
    ax.set_ylabel('Index Size (MB)')
    ax.set_title('Query Latency vs Index Size')
    ax.grid(alpha=0.3)
    
    # Trade-off: Query time vs Throughput
    ax = fig.add_subplot(gs[2, 1])
    times = [c['avg_latency'] for c in all_configs]
    throughputs = [c['throughput'] for c in all_configs]
    colors = ['red' if 'ES' in c['identifier'] else 'coral' for c in all_configs]
    ax.scatter(times, throughputs, alpha=0.6, s=50, c=colors)
    ax.set_xlabel('Query Latency (seconds)')
    ax.set_ylabel('Throughput (queries/sec)')
    ax.set_title('Query Latency vs Throughput')
    ax.grid(alpha=0.3)
    
    # Trade-off: Index size vs Memory
    ax = fig.add_subplot(gs[2, 2])
    sizes = [c['disk_size_mb'] for c in all_configs]
    memory = [c['memory_mb'] for c in all_configs]
    colors = ['red' if 'ES' in c['identifier'] else 'seagreen' for c in all_configs]
    ax.scatter(sizes, memory, alpha=0.6, s=50, c=colors)
    ax.set_xlabel('Index Size (MB)')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Index Size vs Memory Usage')
    ax.grid(alpha=0.3)
    
    plt.savefig(plots_dir / 'comparison_summary.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: comparison_summary.png")
    plt.close()
    
    print("\n" + "="*60)
    print("✓ All plots regenerated successfully!")
    print("="*60)

if __name__ == "__main__":
    check_and_plot()