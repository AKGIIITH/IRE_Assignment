import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class PlotGenerator:
    """Generate plots from benchmark results."""
    
    def __init__(self, results_file: str = "benchmark_results.json"):
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.output_dir = Path("plots")
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_c_index_types(self):
        """Plot.C: Disk size vs Index type (x=1,2,3)."""
        # Filter results for x variations
        x_results = [r for r in self.results if 'i1' in r['identifier'] or 
                     'i2' in r['identifier'] or 'i3' in r['identifier']]
        
        if not x_results:
            print("No x-variant results found for Plot.C")
            return
        
        labels = []
        sizes = []
        
        for r in x_results:
            if 'i1' in r['identifier']:
                labels.append('Boolean (x=1)')
            elif 'i2' in r['identifier']:
                labels.append('WordCount (x=2)')
            elif 'i3' in r['identifier']:
                labels.append('TF-IDF (x=3)')
            else:
                continue
            sizes.append(r['disk_size_mb'])
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, sizes, color=['#3498db', '#e74c3c', '#2ecc71'])
        plt.xlabel('Index Type')
        plt.ylabel('Disk Size (MB)')
        plt.title('Plot.C: Memory Footprint vs Index Type')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plot_c_index_types.png', dpi=300)
        plt.close()
        print("Generated: plot_c_index_types.png")
    
    def plot_a_datastores(self):
        """Plot.A: Latency vs Datastore (y=1,2)."""
        # Filter for datastore variations
        y_results = [r for r in self.results if 'SelfIndex' in r['identifier']]
        
        labels = []
        p95_latencies = []
        
        for r in y_results:
            if 'd1' in r['identifier']:
                labels.append('Custom (y=1)')
            elif 'd2' in r['identifier']:
                labels.append('DB1 (y=2)')
            elif 'd3' in r['identifier']:
                labels.append('DB2 (y=2)')
            else:
                continue
            p95_latencies.append(r['p95_latency'] * 1000)  # Convert to ms
        
        if not labels:
            print("Using available results for Plot.A")
            labels = [r['identifier'][:20] for r in y_results[:3]]
            p95_latencies = [r['p95_latency'] * 1000 for r in y_results[:3]]
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, p95_latencies, color=['#9b59b6', '#f39c12', '#1abc9c'])
        plt.xlabel('Datastore')
        plt.ylabel('P95 Latency (ms)')
        plt.title('Plot.A: Query Latency vs Datastore')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plot_a_datastores.png', dpi=300)
        plt.close()
        print("Generated: plot_a_datastores.png")
    
    def plot_ab_compression(self):
        """Plot.AB: Latency and Throughput vs Compression (z=0,1,2)."""
        # Filter for compression variations
        z_results = []
        for r in self.results:
            if 'c1' in r['identifier']:  # NONE
                z_results.append(('None (z=0)', r))
            elif 'c2' in r['identifier']:  # CODE
                z_results.append(('VByte (z=1)', r))
            elif 'c3' in r['identifier']:  # CLIB
                z_results.append(('Zlib (z=2)', r))
        
        if not z_results:
            print("No compression variants found for Plot.AB")
            return
        
        labels = [label for label, _ in z_results]
        latencies = [r['p95_latency'] * 1000 for _, r in z_results]
        throughputs = [r['throughput'] for _, r in z_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Latency plot
        ax1.bar(labels, latencies, color=['#e74c3c', '#3498db', '#2ecc71'])
        ax1.set_xlabel('Compression Method')
        ax1.set_ylabel('P95 Latency (ms)')
        ax1.set_title('Latency vs Compression')
        ax1.tick_params(axis='x', rotation=15)
        
        # Throughput plot
        ax2.bar(labels, throughputs, color=['#e74c3c', '#3498db', '#2ecc71'])
        ax2.set_xlabel('Compression Method')
        ax2.set_ylabel('Throughput (queries/sec)')
        ax2.set_title('Throughput vs Compression')
        ax2.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plot_ab_compression.png', dpi=300)
        plt.close()
        print("Generated: plot_ab_compression.png")
    
    def plot_a_optimization(self):
        """Plot.A: Latency vs Optimization (i=0,1)."""
        # Find results with and without skip pointers
        opt_results = []
        for r in self.results:
            if 'o0' in r['identifier']:
                opt_results.append(('No Skip (i=0)', r))
            elif 'osp' in r['identifier']:
                opt_results.append(('Skip Pointers (i=1)', r))
        
        if not opt_results:
            print("No optimization variants found")
            return
        
        labels = [label for label, _ in opt_results]
        latencies = [r['p95_latency'] * 1000 for _, r in opt_results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, latencies, color=['#95a5a6', '#27ae60'])
        plt.xlabel('Optimization')
        plt.ylabel('P95 Latency (ms)')
        plt.title('Plot.A: Latency vs Skip Pointer Optimization')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plot_a_optimization.png', dpi=300)
        plt.close()
        print("Generated: plot_a_optimization.png")
    
    def plot_ac_query_processing(self):
        """Plot.AC: Latency and Memory vs Query Processing (q=T,D)."""
        # Filter for query processing variations
        q_results = []
        for r in self.results:
            if 'qT' in r['identifier']:
                q_results.append(('TAAT (q=T)', r))
            elif 'qD' in r['identifier']:
                q_results.append(('DAAT (q=D)', r))
        
        if not q_results:
            print("No query processing variants found")
            return
        
        labels = [label for label, _ in q_results]
        latencies = [r['p95_latency'] * 1000 for _, r in q_results]
        memories = [r['memory_mb'] for _, r in q_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Latency plot
        ax1.bar(labels, latencies, color=['#3498db', '#e74c3c'])
        ax1.set_xlabel('Query Processing Method')
        ax1.set_ylabel('P95 Latency (ms)')
        ax1.set_title('Latency vs Query Processing')
        
        # Memory plot
        ax2.bar(labels, memories, color=['#3498db', '#e74c3c'])
        ax2.set_xlabel('Query Processing Method')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory vs Query Processing')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plot_ac_query_processing.png', dpi=300)
        plt.close()
        print("Generated: plot_ac_query_processing.png")
    
    def plot_comparison_summary(self):
        """Generate overall comparison plot."""
        if len(self.results) < 2:
            print("Not enough results for comparison")
            return
        
        labels = [r['identifier'][:15] for r in self.results[:6]]
        p95_latencies = [r['p95_latency'] * 1000 for r in self.results[:6]]
        throughputs = [r['throughput'] for r in self.results[:6]]
        disk_sizes = [r['disk_size_mb'] for r in self.results[:6]]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # P95 Latency
        axes[0, 0].bar(range(len(labels)), p95_latencies)
        axes[0, 0].set_xticks(range(len(labels)))
        axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, 0].set_ylabel('P95 Latency (ms)')
        axes[0, 0].set_title('P95 Query Latency')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Throughput
        axes[0, 1].bar(range(len(labels)), throughputs, color='green')
        axes[0, 1].set_xticks(range(len(labels)))
        axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Queries/sec')
        axes[0, 1].set_title('Query Throughput')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Disk Size
        axes[1, 0].bar(range(len(labels)), disk_sizes, color='orange')
        axes[1, 0].set_xticks(range(len(labels)))
        axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Disk Size (MB)')
        axes[1, 0].set_title('Index Memory Footprint')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Summary table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table_data = []
        for r in self.results[:6]:
            table_data.append([
                r['identifier'][:15],
                f"{r['p95_latency']*1000:.1f}",
                f"{r['throughput']:.1f}",
                f"{r['disk_size_mb']:.1f}"
            ])
        
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Config', 'P95(ms)', 'QPS', 'Size(MB)'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_summary.png', dpi=300)
        plt.close()
        print("Generated: comparison_summary.png")
    
    def generate_all_plots(self):
        """Generate all required plots."""
        print("Generating all plots...")
        self.plot_c_index_types()
        self.plot_a_datastores()
        self.plot_ab_compression()
        self.plot_a_optimization()
        self.plot_ac_query_processing()
        self.plot_comparison_summary()
        print(f"\nAll plots saved to {self.output_dir}/")

if __name__ == "__main__":
    try:
        plotter = PlotGenerator()
        plotter.generate_all_plots()
    except FileNotFoundError:
        print("benchmark_results.json not found. Run benchmark.py first.")