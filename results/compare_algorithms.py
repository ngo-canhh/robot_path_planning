import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def load_metrics_data(algorithm_name):
    """
    Load metrics data for a specific algorithm from its results directory.
    
    Args:
        algorithm_name: Name of the algorithm (rrt, bb_orca, raytracing_waitingrule)
    
    Returns:
        Dictionary containing all metrics data
    """
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"results/map_results_{algorithm_name}")
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory not found for {algorithm_name}: {results_dir}")
        return None
    
    # Dictionary to store metrics by scenario
    all_scenario_metrics = {}
    
    # Load scenario metrics files
    scenario_types = ["maze", "dense", "room", "trap", "real"]
    
    for scenario_type in scenario_types:
        scenario_file = f"{results_dir}/{scenario_type}_metrics.pkl"
        
        if os.path.exists(scenario_file):
            with open(scenario_file, 'rb') as f:
                scenario_data = pickle.load(f)
                all_scenario_metrics[scenario_type] = scenario_data
    
    return all_scenario_metrics

def load_traditional_metrics_data(algorithm_name):
    """
    Load metrics data for an algorithm from traditional text format.
    
    Args:
        algorithm_name: Name of the algorithm (e.g., 'Quad Dstar Fuzzy')
    
    Returns:
        Dictionary containing all metrics data in a format compatible with other functions
    """
    results_dir = os.path.join(os.path.dirname(__file__), "result_another_algorithms")
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return None
    
    # Dictionary to store metrics by scenario
    all_scenario_metrics = {}
    
    # Load scenario metrics files
    scenario_types = ["maze", "dense", "room", "trap"]
    
    for scenario_type in scenario_types:
        scenario_path = f"{results_dir}/{scenario_type}/{algorithm_name}"
        
        if os.path.exists(scenario_path):
            # Dictionary to store metrics for each map in this scenario
            scenario_data = {}
            
            # Read the file line by line
            with open(scenario_path, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("-----"):
                        continue
                    
                    # Parse the line
                    parts = line.split(":")
                    if len(parts) != 2:
                        continue
                    
                    map_name = parts[0].strip()
                    metrics_str = parts[1].strip()
                    
                    # Check if the test failed
                    failed = "Fail" in metrics_str
                    metrics_str = metrics_str.replace(" Fail", "")
                    
                    # Split metrics values
                    metrics_values = metrics_str.split()
                    
                    # Need at least path_length, smoothness, and time
                    if len(metrics_values) < 3:
                        continue
                    
                    # Parse metrics values
                    path_length = float(metrics_values[0])
                    path_smoothness = float(metrics_values[1])
                    time = float(metrics_values[2])
                    
                    # Parse SPL and SPS if available
                    spl = float(metrics_values[3]) if len(metrics_values) > 3 else 0.0
                    sps = float(metrics_values[4]) if len(metrics_values) > 4 else 0.0
                    
                    # Create metrics dictionary for this map
                    if map_name not in scenario_data:
                        scenario_data[map_name] = {
                            'path_metrics_list': [],
                            'success_status': []
                        }
                    
                    # Add metrics for this run
                    metrics = {
                        'path_length': path_length,
                        'path_smoothness': path_smoothness,
                        'time': time,
                        'spl': spl,
                        'sps': sps
                    }
                    
                    scenario_data[map_name]['path_metrics_list'].append(metrics)
                    scenario_data[map_name]['success_status'].append(not failed)
            
            # Add scenario data to all metrics
            if scenario_data:
                all_scenario_metrics[scenario_type] = scenario_data
    
    return all_scenario_metrics

def calculate_aggregated_metrics(algorithm_metrics):
    """
    Calculate aggregated metrics for each scenario from raw metrics data.
    
    Args:
        algorithm_metrics: Dictionary containing metrics data for an algorithm
        
    Returns:
        Dictionary with aggregated metrics by scenario
    """
    aggregated_metrics = {}
    
    for scenario_name, scenario_data in algorithm_metrics.items():
        # Initialize metrics for this scenario
        success_rates = []
        path_lengths = []
        path_smoothness = []
        spl_values = []
        sps_values = []
        
        # Process each map in the scenario
        for map_name, map_data in scenario_data.items():
            metrics_list = map_data['path_metrics_list']
            success_status = map_data['success_status']
            
            if not metrics_list:
                continue
            
            # Calculate success rate for this map
            success_rate = sum(1 for s in success_status if s) / len(success_status) * 100
            success_rates.append(success_rate)
            
            # Calculate average metrics for this map
            avg_path_length = np.mean([m['path_length'] for m in metrics_list])
            avg_smoothness = np.mean([m['path_smoothness'] for m in metrics_list])
            avg_spl = np.mean([m['spl'] for m in metrics_list])
            avg_sps = np.mean([m['sps'] for m in metrics_list])
            
            path_lengths.append(avg_path_length)
            path_smoothness.append(avg_smoothness)
            spl_values.append(avg_spl)
            sps_values.append(avg_sps)
        
        # Calculate scenario averages
        if success_rates:
            aggregated_metrics[scenario_name] = {
                'success_rate': np.mean(success_rates ),
                'path_length': np.mean(path_lengths),
                'path_smoothness': np.mean(path_smoothness),
                'spl': np.mean(spl_values),
                'sps': np.mean(sps_values)
            }
    
    return aggregated_metrics

def calculate_overall_metrics(aggregated_metrics):
    """
    Calculate overall metrics across all scenarios.
    
    Args:
        aggregated_metrics: Dictionary with aggregated metrics by scenario
        
    Returns:
        Dictionary with overall metrics
    """
    if not aggregated_metrics:
        return None
    
    success_rates = []
    path_lengths = []
    smoothness = []
    spl_values = []
    sps_values = []
    
    for scenario_metrics in aggregated_metrics.values():
        success_rates.append(scenario_metrics['success_rate'])
        path_lengths.append(scenario_metrics['path_length'])
        smoothness.append(scenario_metrics['path_smoothness'])
        spl_values.append(scenario_metrics['spl'])
        sps_values.append(scenario_metrics['sps'])
    
    return {
        'success_rate': np.mean(success_rates),
        'path_length': np.mean(path_lengths),
        'path_smoothness': np.mean(smoothness),
        'spl': np.mean(spl_values),
        'sps': np.mean(sps_values)
    }

def create_comparison_bar_charts(algorithms_data, output_dir):
    """
    Create bar charts comparing algorithms for each metric and scenario.
    
    Args:
        algorithms_data: Dictionary with metrics for each algorithm
        output_dir: Directory to save output visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of scenarios (assuming all algorithms have the same scenarios)
    scenarios = list(next(iter(algorithms_data.values())).keys())
    algorithm_names = list(algorithms_data.keys())
    
    # Define metrics to compare
    metrics = [
        ('success_rate', 'Success Rate (%)', 0, 100),
        ('path_length', 'Path Length', None, None),
        ('path_smoothness', 'Path Smoothness', 0, 1),
        ('spl', 'SPL', 0, 1),
        ('sps', 'SPS', 0, 1)
    ]
    
    # Create a figure for each metric
    for metric_key, metric_name, y_min, y_max in metrics:
        plt.figure(figsize=(14, 8))
        plt.suptitle(f'Comparison of {metric_name} Across Algorithms and Scenarios', fontsize=16)
        
        # Bar positions
        x = np.arange(len(scenarios))
        width = 0.2  # Width of bars - adjusted for more algorithms
        
        # Plot bars for each algorithm
        for i, (alg_name, alg_data) in enumerate(algorithms_data.items()):
            # Filter scenarios that exist in this algorithm's data
            valid_scenarios = [s for s in scenarios if s in alg_data]
            if not valid_scenarios:
                continue
                
            values = [alg_data[scenario][metric_key] if scenario in alg_data else 0 for scenario in scenarios]
            plt.bar(x + (i - len(algorithms_data)/2 + 0.5) * width, values, width, label=alg_name)
        
        # Add labels and legend
        plt.xlabel('Scenario')
        plt.ylabel(metric_name)
        plt.xticks(x, scenarios)
        plt.legend()
        
        # Set y-limits if specified
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        
        # Add grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_{metric_key}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create single figure with all metrics
    plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2)
    
    for i, (metric_key, metric_name, y_min, y_max) in enumerate(metrics):
        ax = plt.subplot(gs[i])
        
        for j, (alg_name, alg_data) in enumerate(algorithms_data.items()):
            values = [alg_data[scenario][metric_key] if scenario in alg_data else 0 for scenario in scenarios]
            ax.bar(x + (j - len(algorithms_data)/2 + 0.5) * width, values, width, label=alg_name)
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Only show legend on the first plot
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/all_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_radar_chart(algorithms_data, output_dir):
    """
    Create radar chart comparing overall algorithm performance.
    
    Args:
        algorithms_data: Dictionary with overall metrics for each algorithm
        output_dir: Directory to save output visualizations
    """
    # Prepare data for radar chart
    categories = ['Success Rate', 'Path Efficiency', 'Path Smoothness', 'SPL', 'SPS']
    
    # Normalize data for radar chart (all values between 0 and 1)
    normalized_data = {}
    
    for alg_name, metrics in algorithms_data.items():
        normalized_data[alg_name] = [
            metrics['success_rate'] / 100,  # Success rate (percentage to 0-1)
            1.0 / (metrics['path_length'] / 700),  # Path length (inverse and normalized)
            metrics['path_smoothness'],  # Already between 0-1
            metrics['spl'],  # Already between 0-1
            metrics['sps']   # Already between 0-1
        ]
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(categories)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    # Plot for each algorithm
    for alg_name, values in normalized_data.items():
        values += values[:1]  # Close the polygon
        ax.plot(angles, values, linewidth=2, label=alg_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Algorithm Comparison - Overall Performance', size=15)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/radar_chart_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_overall_bar_chart(algorithms_data, output_dir):
    """
    Create bar chart comparing overall algorithm performance.
    
    Args:
        algorithms_data: Dictionary with overall metrics for each algorithm
        output_dir: Directory to save output visualizations
    """
    # Define metrics to compare
    metrics = [
        ('success_rate', 'Success Rate (%)', 0, 100),
        ('path_length', 'Path Length', None, None),
        ('path_smoothness', 'Path Smoothness', 0, 1),
        ('spl', 'SPL', 0, 1),
        ('sps', 'SPS', 0, 1)
    ]
    
    alg_names = list(algorithms_data.keys())
    
    plt.figure(figsize=(14, 10))
    
    # Create subplots for each metric
    for i, (metric_key, metric_name, y_min, y_max) in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        
        values = [algorithms_data[alg_name][metric_key] for alg_name in alg_names]
        
        bars = plt.bar(alg_names, values)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.title(metric_name)
        
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle('Overall Algorithm Performance Comparison', fontsize=16, y=1.02)
    plt.savefig(f"{output_dir}/overall_bar_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to run the algorithm comparison.
    """
    print("Starting algorithm comparison...")
    
    # Define algorithms to compare
    algorithms = ['rrt', 'bb_orca', 'raytracing_waitingrule'] #, 'Quad Dstar Fuzzy']
    
    # Output directory for comparison visualizations
    output_dir = os.path.join(os.path.dirname(__file__), "algorithm_comparison_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data for each algorithm
    all_algorithms_data = {}
    overall_metrics = {}
    
    for algorithm in algorithms:
        print(f"Loading data for algorithm: {algorithm}")
        
        # Check if this is a traditional format algorithm
        if algorithm == 'Quad Dstar Fuzzy':
            metrics_data = load_traditional_metrics_data(algorithm)
        else:
            metrics_data = load_metrics_data(algorithm)
        
        if metrics_data:
            # Calculate aggregated metrics by scenario
            aggregated_metrics = calculate_aggregated_metrics(metrics_data)
            all_algorithms_data[algorithm] = aggregated_metrics
            
            # Calculate overall metrics
            overall_metrics[algorithm] = calculate_overall_metrics(aggregated_metrics)
            
            print(f"  Successfully loaded metrics for {algorithm}")
        else:
            print(f"  Failed to load metrics for {algorithm}")
    
    # Create visualizations
    if all_algorithms_data:
        print("Creating comparison visualizations...")
        
        # Create bar charts comparing metrics across scenarios
        create_comparison_bar_charts(all_algorithms_data, output_dir)
        
        # Create radar chart for overall comparison
        create_radar_chart(overall_metrics, output_dir)
        
        # Create overall bar chart comparison
        create_overall_bar_chart(overall_metrics, output_dir)
        
        print(f"Visualizations saved to {output_dir}/")
    else:
        print("No data available for comparison. Please run the algorithms first.")

if __name__ == "__main__":
    main() 