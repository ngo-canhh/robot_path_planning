import numpy as np
import matplotlib.pyplot as plt

def print_metrics(metrics):
    """
    Print the path planning metrics in a formatted way
    
    Args:
        metrics (dict): The metrics dictionary from get_path_metrics()
    """
    print("===== Path Planning Metrics =====")
    print(f"Path Length: {metrics['path_length']:.2f} units")
    print(f"Path Smoothness: {metrics['path_smoothness']:.4f} radians")
    print(f"SPL (Success weighted by Path Length): {metrics['spl']:.4f}")
    print(f"SPS (Success weighted by Path Smoothness): {metrics['sps']:.4f}")
    print(f"Success: {'Yes' if metrics['success'] else 'No'}")
    print("================================")

def visualize_metrics(metrics_list, output_file=None, display=False, title=None):
    """
    Visualize a list of metrics from multiple episodes
    
    Args:
        metrics_list (list): List of metrics dictionaries from get_path_metrics()
        output_file (str): Path to save the visualization. If None, won't save.
        display (bool): Whether to display the figure using plt.show()
        title (str): Custom title for the figure. If None, no title is added.
    """
    if not metrics_list:
        print("No metrics data available for visualization")
        return
    
    # Debug information
    print(f"Visualizing metrics for {len(metrics_list)} episodes")
    
    # Extract the data
    path_lengths = [m['path_length'] for m in metrics_list]
    smoothness = [m['path_smoothness'] for m in metrics_list]
    spls = [m['spl'] for m in metrics_list]
    spss = [m['sps'] for m in metrics_list]
    success = [m['success'] for m in metrics_list]
    
    # Use actual number of episodes for x-axis
    episodes = list(range(1, len(metrics_list) + 1))
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot path length
    axs[0, 0].plot(episodes, path_lengths, 'b-o', label='Path Length')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Units')
    axs[0, 0].set_title('Path Length vs Episode')
    axs[0, 0].grid(True)
    
    # Set x-axis ticks to be integer values only
    axs[0, 0].set_xticks(episodes)
    axs[0, 0].set_xlim(0.5, len(episodes) + 0.5)
    
    # Plot success rate (shown as binary 0/1)
    axs[0, 1].bar(episodes, success, color='g', label='Success (0/1)')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Success (1=Yes, 0=No)')
    axs[0, 1].set_title('Success vs Episode')
    axs[0, 1].grid(True)
    axs[0, 1].set_xticks(episodes)
    axs[0, 1].set_ylim(-0.1, 1.1)
    axs[0, 1].set_xlim(0.5, len(episodes) + 0.5)
    
    # Plot smoothness
    axs[1, 0].plot(episodes, smoothness, 'r-o', label='Path Smoothness')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Angle (radians)')
    axs[1, 0].set_title('Path Smoothness vs Episode')
    axs[1, 0].grid(True)
    axs[1, 0].set_xticks(episodes)
    axs[1, 0].set_xlim(0.5, len(episodes) + 0.5)
    
    # Plot SPL and SPS
    axs[1, 1].plot(episodes, spls, 'c-o', label='SPL')
    axs[1, 1].plot(episodes, spss, 'm-o', label='SPS')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Metric Value')
    axs[1, 1].set_title('SPL and SPS vs Episode')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    axs[1, 1].set_xticks(episodes)
    axs[1, 1].set_xlim(0.5, len(episodes) + 0.5)
    axs[1, 1].set_ylim(-0.1, 1.1)  # SPL and SPS are between 0 and 1
    
    # Add success markers
    for i, s in enumerate(success):
        for ax in axs.flatten():
            if s:
                ax.axvspan(i+0.5, i+1.5, alpha=0.2, color='green')
            else:
                ax.axvspan(i+0.5, i+1.5, alpha=0.2, color='red')
    
    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    # Show only if display is True
    if display:
        plt.show()
    else:
        plt.close(fig)  # Close the figure if not displaying

def radar_chart(metrics, output_file=None, display=False, title='Path Planning Performance Radar Chart'):
    """
    Create a radar chart visualization for a single set of metrics
    
    Args:
        metrics (dict): The metrics dictionary from get_path_metrics()
        output_file (str): Path to save the visualization. If None, won't save.
        display (bool): Whether to display the figure using plt.show()
        title (str): Custom title for the radar chart
    """
    # Categories for radar chart based on our key metrics
    categories = ['Path Efficiency', 'Path Smoothness', 'Success', 'SPL', 'SPS']
    
    # Calculate normalized values (0-1 scale)
    # For path efficiency, we use SPL which already captures this concept
    path_efficiency = 1.0 / (metrics['path_length'] / 700)
    
    # For smoothness, lower is better (closer to 0 radians means straighter path)
    # Normalize to 0-1 where 1 is best (smoothest)
    smoothness_norm = 1.0 - min(metrics['path_smoothness'] / (np.pi/2), 1.0)
    
    # Success is already binary (0/1)
    success = metrics['success']
    
    values = [
        path_efficiency,
        smoothness_norm,
        success,
        metrics['spl'],
        metrics['sps']
    ]
    
    # Repeat the first value to close the polygon
    values += [values[0]]
    categories += [categories[0]]
    
    # Calculate angle for each category
    angles = [n / float(len(categories)-1) * 2 * np.pi for n in range(len(categories))]
    
    # Create the plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Draw the polygon
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])
    
    # Set radial ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_ylim(0, 1)
    
    # Add title
    plt.title(title, size=15, y=1.1)
    
    plt.tight_layout()
    
    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    # Show only if display is True
    if display:
        plt.show()
    else:
        plt.close(fig)  # Close the figure if not displaying

# Example usage:
if __name__ == "__main__":
    # Example metrics data
    example_metrics = {
        'path_length': 150.5,
        'path_smoothness': 0.15,
        'spl': 0.85,
        'sps': 0.78,
        'success': 1
    }
    
    print_metrics(example_metrics)
    
    # Example of multiple metrics for visualization
    metrics_list = [
        example_metrics,
        {
            'path_length': 180.3,
            'path_smoothness': 0.25,
            'spl': 0.72,
            'sps': 0.65,
            'success': 1
        },
        {
            'path_length': 200.1,
            'path_smoothness': 0.45,
            'spl': 0.60,
            'sps': 0.52,
            'success': 0
        }
    ]
    
    # When running standalone, display the figures
    visualize_metrics(metrics_list, output_file='path_planning_metrics.png', display=True)
    radar_chart(example_metrics, output_file='path_performance_radar.png', display=True) 