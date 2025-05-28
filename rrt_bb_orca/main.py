from indoor_robot_env import IndoorRobotEnv
from components.obstacle import StaticObstacle, DynamicObstacle
from components.shape import Circle, Rectangle, Triangle, Polygon
from components.maps import get_empty_map
from components.maps import maps
import matplotlib.pyplot as plt
import numpy as np
import math
from metrics_display import print_metrics, visualize_metrics, radar_chart
import os
import pickle
import importlib
import sys
import time

BASE_SEED = 88

# Define metrics callback function
def metrics_callback(metrics):
    """Callback function to process metrics from the environment"""
    print(f"Metrics update:")
    print(f"  Path length: {metrics['path_length']:.2f}")
    print(f"  Path smoothness: {metrics['path_smoothness']:.4f}")
    print(f"  Success: {metrics['success']}")
    print(f"  SPL: {metrics['spl']:.4f}")
    print(f"  SPS: {metrics['sps']:.4f}")

# Hàm để lấy controller dựa trên lựa chọn thuật toán
def get_controller_class(algorithm):
    """
    Dynamically import and return the controller class based on algorithm choice
    
    Args:
        algorithm: String name of the algorithm ('rrt', 'bb_orca', 'raytracing_waitingrule')
    
    Returns:
        The controller class to use
    """
    if algorithm == 'rrt':
        module_name = 'indoor_robot_controller_rrt'
    elif algorithm == 'bb_orca':
        module_name = 'indoor_robot_controller_bb_orca'
    elif algorithm == 'raytracing_waitingrule':
        module_name = 'indoor_robot_controller_raytracing_waitingrule'
    else:
        print(f"Không nhận diện được thuật toán: {algorithm}")
        print("Sử dụng thuật toán mặc định: bb_orca")
        module_name = 'indoor_robot_controller_bb_orca'
    
    try:
        module = importlib.import_module(module_name)
        return module.IndoorRobotController
    except ImportError as e:
        print(f"Lỗi khi import module {module_name}: {e}")
        print("Sử dụng fallback: indoor_robot_controller_bb_orca")
        from indoor_robot_controller_bb_orca import IndoorRobotController
        return IndoorRobotController

# Hàm tương tác với người dùng để nhập tham số
def get_user_input():
    """
    Hàm tương tác với người dùng trong terminal để lấy các tham số đầu vào
    
    Returns:
        dictionary chứa các tham số đã chọn
    """
    # Cấu hình mặc định
    config = {
        'algorithm': 'bb_orca',
        'scenario': 'all',
        'episodes': 3,
        'render': 'human'
    }
    
    # Hiển thị menu
    print("\n===== CHƯƠNG TRÌNH MÔ PHỎNG ROBOT =====")
    print("Vui lòng chọn các tham số sau:")
    
    # Chọn thuật toán
    print("\n-- THUẬT TOÁN --")
    print("1. RRT")
    print("2. BB-ORCA (mặc định)")
    print("3. Raytracing-Waitingrule")
    
    while True:
        choice = input("Lựa chọn của bạn (1-3, Enter để chọn mặc định): ").strip()
        if choice == "":
            break  # Giữ mặc định
        elif choice == "1":
            config['algorithm'] = 'rrt'
            break
        elif choice == "2":
            config['algorithm'] = 'bb_orca'
            break
        elif choice == "3":
            config['algorithm'] = 'raytracing_waitingrule'
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng thử lại!")
    
    # Chọn scenario
    print("\n-- SCENARIO --")
    print("1. Maze")
    print("2. Dense")
    print("3. Room")
    print("4. Trap")
    print("5. Real")
    print("6. Tất cả (mặc định)")
    
    while True:
        choice = input("Lựa chọn của bạn (1-6, Enter để chọn mặc định): ").strip()
        if choice == "":
            break  # Giữ mặc định
        elif choice == "1":
            config['scenario'] = 'maze'
            break
        elif choice == "2":
            config['scenario'] = 'dense'
            break
        elif choice == "3":
            config['scenario'] = 'room'
            break
        elif choice == "4":
            config['scenario'] = 'trap'
            break
        elif choice == "5":
            config['scenario'] = 'real'
            break
        elif choice == "6":
            config['scenario'] = 'all'
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng thử lại!")
    
    # Nhập số lượng episode
    while True:
        episodes = input("\nSố lượng episode cho mỗi map (Enter để chọn mặc định=3): ").strip()
        if episodes == "":
            break  # Giữ mặc định
        try:
            episodes_num = int(episodes)
            if episodes_num > 0:
                config['episodes'] = episodes_num
                break
            else:
                print("Số episode phải lớn hơn 0!")
        except ValueError:
            print("Vui lòng nhập một số nguyên hợp lệ!")
    
    # Chọn chế độ render
    print("\n-- CHẾ ĐỘ RENDER --")
    print("1. Human - hiển thị giao diện (mặc định)")
    print("2. RGB Array - chạy nhanh không hiển thị")
    
    while True:
        choice = input("Lựa chọn của bạn (1-2, Enter để chọn mặc định): ").strip()
        if choice == "":
            break  # Giữ mặc định
        elif choice == "1":
            config['render'] = 'human'
            break
        elif choice == "2":
            config['render'] = 'rgb_array'
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng thử lại!")
    
    # Xác nhận tham số
    print("\n=== THÔNG SỐ ĐÃ CHỌN ===")
    print(f"Thuật toán: {config['algorithm']}")
    print(f"Scenario: {config['scenario']}")
    print(f"Số episode: {config['episodes']}")
    print(f"Chế độ render: {config['render']}")
    
    confirm = input("\nXác nhận và tiếp tục? (y/n, Enter=y): ").strip().lower()
    if confirm == "" or confirm == "y":
        return config
    else:
        print("Hủy bỏ. Thoát chương trình...")
        sys.exit(0)

# Main simulation loop
if __name__ == "__main__":
    # Lấy tham số từ người dùng
    config = get_user_input()
    
    # Lấy controller class dựa trên thuật toán được chọn
    ControllerClass = get_controller_class(config['algorithm'])
    algorithm_name = config['algorithm']
    
    # Create a directory for saving metrics
    results_dir = f"map_results_{config['algorithm']}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Determine which map types to run
    if config['scenario'].lower() == 'all':
        map_types = ["maze", "dense", "room", "trap", "real"]
    else:
        # Confirm the scenario exists
        scenario = config['scenario'].lower()
        valid_scenarios = ["maze", "dense", "room", "trap", "real"]
        if scenario not in valid_scenarios:
            print(f"Scenario không hợp lệ: {scenario}")
            print(f"Các scenario hợp lệ: {', '.join(valid_scenarios)}")
            sys.exit(1)
        map_types = [scenario]
    
    # Configure simulation parameters
    n_episodes = config['episodes']  # Number of episodes per map
    max_steps = 500  # Maximum steps per episode
    render_mode = config['render']  # Use the specified render mode
    
    # Start and goal positions
    start_pos = (64, 500)
    goal_pos = (470, 180)
    
    # Dictionary to store all results for final reporting
    all_scenario_results = {}
    
    # Loop through all map types
    for map_type in map_types:
        print(f"\n=== Testing map type: {map_type} ===")
        
        # Find all maps of this type
        map_names = [key for key in maps.keys() if key.startswith(map_type)]
        
        # Sort map indices by number
        map_names.sort(key=lambda x: int(x[len(map_type):]) if x[len(map_type):].isdigit() else 0)
        
        # Dictionary to store metrics for all maps of this type
        all_map_metrics = {}
        
        # Loop through each map of this type
        for map_name in map_names:
            print(f"\n--- Running map: {map_name} ---")
            
            # Initialize empty lists for storing metrics for this map
            # Make sure we're creating fresh lists for each map
            all_metrics = []
            path_metrics_list = []
            success_status = []
            
            def store_metrics_callback(metrics):
                """Store a copy of the metrics for later analysis"""
                all_metrics.append(metrics.copy())
                
                # Also store the path metrics in our simplified format
                if hasattr(env, 'get_path_metrics'):
                    path_metrics = env.get_path_metrics()
                    path_metrics_list.append(path_metrics)
            
            # Create environment with this map
            env = IndoorRobotEnv(
                robot_radius=3,
                max_steps=max_steps,
                sensor_range=50, 
                render_mode=render_mode,
                config_path=get_empty_map(530),
                metrics_callback=store_metrics_callback,
            )
            
            # Add obstacles from map
            env.add_obstacles(maps[map_name])
            
            # Set specific start and goal positions
            env.start = start_pos
            env.goal = goal_pos
            
            # Create controller instance with the chosen class
            controller = ControllerClass(env)
            controller.add_static_obstacles(env.vanilla_obstacles)
            
            # Run episodes for this map
            for episode in range(n_episodes):
                observation, info = env.reset(seed=BASE_SEED + episode)
                controller.reset()
                print(f"\nStarting episode {episode+1} for map {map_name}")
                
                terminated = False
                truncated = False
                
                # Step loop
                while not terminated and not truncated:
                    # Get action from controller
                    action, controller_info = controller.get_action(observation)
                    
                    # Step environment
                    observation, reward, terminated, truncated, info = env.step(action)
                    
                    # Render with controller info
                    env.render(controller_info=controller_info)
                
                # Record episode success
                episode_status = info['status'] if 'status' in info else 'unknown'
                success = episode_status == 'goal_reached'
                success_status.append(success)
                
                print(f"Episode {episode+1} ended with status: {episode_status}")
            
            # Close environment
            env.close()
            
            # Print brief summary for this map (minimal output during runs)
            if all_metrics:
                latest_metrics = all_metrics[-1]
                success_rate = latest_metrics['success'] * 100 if n_episodes == 1 else sum(m['success'] for m in all_metrics) / n_episodes * 100
                print(f"Map {map_name} - Success rate: {success_rate:.1f}%")
            
            # Verify that we have the correct number of metrics (matching number of episodes)
            # Cut off any excess metrics from previous runs if necessary
            if len(path_metrics_list) > n_episodes:
                print(f"Warning: Found {len(path_metrics_list)} metrics records but only ran {n_episodes} episodes.")
                print(f"Trimming metrics list to most recent {n_episodes} episodes.")
                path_metrics_list = path_metrics_list[-n_episodes:]
                all_metrics = all_metrics[-n_episodes:]
                success_status = success_status[-n_episodes:]
                
            # Store metrics for this map
            all_map_metrics[map_name] = {
                'all_metrics': all_metrics,
                'path_metrics_list': path_metrics_list,
                'success_status': success_status
            }
            
            # Save metrics for this specific map to a separate file
            map_results_file = f"{results_dir}/{map_name}_metrics.pkl"
            with open(map_results_file, 'wb') as f:
                pickle.dump({
                    'map_name': map_name,
                    'all_metrics': all_metrics,
                    'path_metrics_list': path_metrics_list,
                    'success_status': success_status
                }, f)
            print(f"Đã lưu kết quả cho map {map_name} vào {map_results_file}")
            
            # Generate visualizations for this map
            if path_metrics_list:
                # Create radar chart for the latest episode
                radar_chart(path_metrics_list[-1], 
                           output_file=f"{results_dir}/{map_name}_radar.png", 
                           display=False,
                           title=f"Map {map_name} Performance ({algorithm_name})")
                
                # Visualize metrics progression if we have multiple episodes
                if len(path_metrics_list) > 1:
                    visualize_metrics(path_metrics_list, 
                                     output_file=f"{results_dir}/{map_name}_episode_metrics.png",
                                     display=False,
                                     title=f"Episode Comparison for {map_name} ({algorithm_name})")
        
        # After processing all maps of this type, create scenario-level comparison
        if len(all_map_metrics.keys()) > 1:
            # Create comparison visualization for maps in this scenario
            plt.figure(figsize=(15, 12))
            plt.suptitle(f"Maps Comparison for Scenario: {map_type} ({algorithm_name})", fontsize=16)
            
            map_names_in_scenario = list(all_map_metrics.keys())
            indices = np.arange(len(map_names_in_scenario))
            width = 0.15
            
            # Plot success rates for all maps in this scenario
            plt.subplot(3, 2, 1)
            success_rates = [sum(all_map_metrics[name]['success_status']) / len(all_map_metrics[name]['success_status']) * 100 
                           for name in map_names_in_scenario]
            plt.bar(indices, success_rates, width=0.6)
            plt.ylabel('Success Rate (%)')
            plt.title('Success Rate by Map')
            plt.xticks(indices, map_names_in_scenario, rotation=45)
            plt.ylim(0, 100)
            
            # Path Length
            plt.subplot(3, 2, 2)
            path_lengths = [np.mean([m['path_length'] for m in all_map_metrics[name]['path_metrics_list']])
                          for name in map_names_in_scenario]
            plt.bar(indices, path_lengths, width=0.6)
            plt.ylabel('Path Length')
            plt.title('Avg Path Length by Map')
            plt.xticks(indices, map_names_in_scenario, rotation=45)
            
            # Path Smoothness
            plt.subplot(3, 2, 3)
            smoothness = [np.mean([m['path_smoothness'] for m in all_map_metrics[name]['path_metrics_list']])
                        for name in map_names_in_scenario]
            plt.bar(indices, smoothness, width=0.6)
            plt.ylabel('Smoothness')
            plt.title('Avg Path Smoothness by Map')
            plt.xticks(indices, map_names_in_scenario, rotation=45)
            
            # SPL
            plt.subplot(3, 2, 4)
            spl_values = [np.mean([m['spl'] for m in all_map_metrics[name]['path_metrics_list']])
                        for name in map_names_in_scenario]
            plt.bar(indices, spl_values, width=0.6)
            plt.ylabel('SPL')
            plt.title('Avg SPL by Map')
            plt.xticks(indices, map_names_in_scenario, rotation=45)
            plt.ylim(0, 1)
            
            # SPS
            plt.subplot(3, 2, 5)
            sps_values = [np.mean([m['sps'] for m in all_map_metrics[name]['path_metrics_list']])
                        for name in map_names_in_scenario]
            plt.bar(indices, sps_values, width=0.6)
            plt.ylabel('SPS')
            plt.title('Avg SPS by Map')
            plt.xticks(indices, map_names_in_scenario, rotation=45)
            plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/{map_type}_maps_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # Save all metrics for this map type
        with open(f"{results_dir}/{map_type}_metrics.pkl", 'wb') as f:
            pickle.dump(all_map_metrics, f)
        
        # Store this scenario's results for final reporting
        all_scenario_results[map_type] = all_map_metrics
    
    # Now that all maps have been processed, display comprehensive metrics
    print("\n\n========== FINAL RESULTS ==========")
    print(f"Algorithm: {algorithm_name}")
    print(f"Scenarios tested: {', '.join(map_types)}")
    
    # Create summary metrics across all scenarios
    all_success_rates = []
    all_path_lengths = []
    all_smoothness = []
    all_spl_values = []
    all_sps_values = []
    
    # Process each scenario's results
    for scenario_name, scenario_data in all_scenario_results.items():
        print(f"\n--- Scenario: {scenario_name} ---")
        scenario_success_rates = []
        scenario_path_lengths = []
        scenario_smoothness = []
        scenario_spl_values = []
        scenario_sps_values = []
        
        # Process each map in this scenario
        for map_name, map_data in scenario_data.items():
            metrics_list = map_data['path_metrics_list']
            success_status = map_data['success_status']
            
            if not metrics_list:
                continue
                
            # Calculate success rate
            success_rate = sum(1 for s in success_status if s) / len(success_status) * 100
            scenario_success_rates.append(success_rate)
            
            # Calculate average metrics for this map
            avg_path_length = np.mean([m['path_length'] for m in metrics_list])
            avg_smoothness = np.mean([m['path_smoothness'] for m in metrics_list])
            avg_spl = np.mean([m['spl'] for m in metrics_list])
            avg_sps = np.mean([m['sps'] for m in metrics_list])
            
            scenario_path_lengths.append(avg_path_length)
            scenario_smoothness.append(avg_smoothness)
            scenario_spl_values.append(avg_spl)
            scenario_sps_values.append(avg_sps)
            
            # Print map results
            print(f"  Map {map_name}:")
            print(f"    Success Rate: {success_rate:.1f}%")
            print(f"    Avg Path Length: {avg_path_length:.2f}")
            print(f"    Avg Path Smoothness: {avg_smoothness:.4f}")
            print(f"    Avg SPL: {avg_spl:.4f}")
            print(f"    Avg SPS: {avg_sps:.4f}")
        
        # Calculate scenario averages
        if scenario_success_rates:
            scenario_avg_success = np.mean(scenario_success_rates)
            scenario_avg_path_length = np.mean(scenario_path_lengths)
            scenario_avg_smoothness = np.mean(scenario_smoothness)
            scenario_avg_spl = np.mean(scenario_spl_values)
            scenario_avg_sps = np.mean(scenario_sps_values)
            
            # Append to all metrics for overall average
            all_success_rates.append(scenario_avg_success)
            all_path_lengths.append(scenario_avg_path_length)
            all_smoothness.append(scenario_avg_smoothness)
            all_spl_values.append(scenario_avg_spl)
            all_sps_values.append(scenario_avg_sps)
            
            # Print scenario averages
            print(f"\n  SCENARIO {scenario_name.upper()} AVERAGE:")
            print(f"    Success Rate: {scenario_avg_success:.1f}%")
            print(f"    Avg Path Length: {scenario_avg_path_length:.2f}")
            print(f"    Avg Path Smoothness: {scenario_avg_smoothness:.4f}")
            print(f"    Avg SPL: {scenario_avg_spl:.4f}")
            print(f"    Avg SPS: {scenario_avg_sps:.4f}")
    
    # Print overall averages
    if all_success_rates:
        overall_success = np.mean(all_success_rates)
        overall_path_length = np.mean(all_path_lengths)
        overall_smoothness = np.mean(all_smoothness)
        overall_spl = np.mean(all_spl_values)
        overall_sps = np.mean(all_sps_values)
        
        print("\n=== OVERALL ALGORITHM PERFORMANCE ===")
        print(f"Algorithm: {algorithm_name}")
        print(f"Overall Success Rate: {overall_success:.1f}%")
        print(f"Overall Avg Path Length: {overall_path_length:.2f}")
        print(f"Overall Avg Path Smoothness: {overall_smoothness:.4f}")
        print(f"Overall Avg SPL: {overall_spl:.4f}")
        print(f"Overall Avg SPS: {overall_sps:.4f}")
        
        # Create comparative visualization across all scenarios
        if len(map_types) > 1:
            # Metrics comparison across scenarios
            plt.figure(figsize=(15, 12))
            
            # Bar indices and width
            indices = np.arange(len(map_types))
            width = 0.15
            
            # Success rates
            plt.subplot(3, 2, 1)
            success_rates_by_scenario = [np.mean([sum(1 for s in scenario_data[map_name]['success_status'] if s) / len(scenario_data[map_name]['success_status']) * 100
                                       for map_name in scenario_data]) for scenario_name, scenario_data in all_scenario_results.items()]
            plt.bar(indices, success_rates_by_scenario, width=0.6)
            plt.ylabel('Success Rate (%)')
            plt.title('Success Rate by Scenario')
            plt.xticks(indices, map_types, rotation=45)
            plt.ylim(0, 100)
            
            # Path Length
            plt.subplot(3, 2, 2)
            path_lengths_by_scenario = [np.mean([np.mean([m['path_length'] for m in scenario_data[map_name]['path_metrics_list']])
                                     for map_name in scenario_data]) for scenario_name, scenario_data in all_scenario_results.items()]
            plt.bar(indices, path_lengths_by_scenario, width=0.6)
            plt.ylabel('Path Length')
            plt.title('Avg Path Length by Scenario')
            plt.xticks(indices, map_types, rotation=45)
            
            # Path Smoothness
            plt.subplot(3, 2, 3)
            smoothness_by_scenario = [np.mean([np.mean([m['path_smoothness'] for m in scenario_data[map_name]['path_metrics_list']])
                                   for map_name in scenario_data]) for scenario_name, scenario_data in all_scenario_results.items()]
            plt.bar(indices, smoothness_by_scenario, width=0.6)
            plt.ylabel('Smoothness')
            plt.title('Avg Path Smoothness by Scenario')
            plt.xticks(indices, map_types, rotation=45)
            
            # SPL
            plt.subplot(3, 2, 4)
            spl_by_scenario = [np.mean([np.mean([m['spl'] for m in scenario_data[map_name]['path_metrics_list']])
                              for map_name in scenario_data]) for scenario_name, scenario_data in all_scenario_results.items()]
            plt.bar(indices, spl_by_scenario, width=0.6)
            plt.ylabel('SPL')
            plt.title('Avg SPL by Scenario')
            plt.xticks(indices, map_types, rotation=45)
            plt.ylim(0, 1)
            
            # SPS
            plt.subplot(3, 2, 5)
            sps_by_scenario = [np.mean([np.mean([m['sps'] for m in scenario_data[map_name]['path_metrics_list']])
                              for map_name in scenario_data]) for scenario_name, scenario_data in all_scenario_results.items()]
            plt.bar(indices, sps_by_scenario, width=0.6)
            plt.ylabel('SPS')
            plt.title('Avg SPS by Scenario')
            plt.xticks(indices, map_types, rotation=45)
            plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.suptitle(f'Algorithm Performance Across Scenarios: {algorithm_name}', fontsize=16, y=1.02)
            plt.savefig(f"{results_dir}/scenario_comparison_bar.png", dpi=300, bbox_inches='tight')
            # plt.show()  # Comentado para evitar mostrar visualizaciones
            
            # Create radar chart for comparing scenarios
            plt.figure(figsize=(12, 10))
            plt.suptitle(f'Algorithm Radar Comparison by Scenario: {algorithm_name}', fontsize=16, y=1.02)
            
            # Prepare averaged metrics for each scenario
            scenario_metrics = {}
            for i, scenario_name in enumerate(map_types):
                scenario_metrics[scenario_name] = {
                    'success_rate': success_rates_by_scenario[i] / 100,  # Convert to 0-1 scale
                    'path_length_inv': 1.0 / (path_lengths_by_scenario[i] if path_lengths_by_scenario[i] > 0 else 1.0),  # Invert so smaller is better
                    'path_smoothness': smoothness_by_scenario[i],
                    'spl': spl_by_scenario[i], 
                    'sps': sps_by_scenario[i]
                }
                
            # Create radar chart comparing scenarios
            if len(map_types) <= 5:  # Only create radar chart if we have a reasonable number of scenarios
                radar_metrics = ['success_rate', 'path_length_inv', 'path_smoothness', 'spl', 'sps']
                radar_labels = ['Success Rate', 'Path Efficiency', 'Path Smoothness', 'SPL', 'SPS']
                
                # Plot radar chart
                from matplotlib.path import Path
                from matplotlib.spines import Spine
                from matplotlib.transforms import Affine2D
                
                angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
                angles += angles[:1]  # Close the loop
                
                ax = plt.subplot(111, polar=True)
                
                # Add lines for each scenario
                for scenario_name in map_types:
                    values = [scenario_metrics[scenario_name][metric] for metric in radar_metrics]
                    values += values[:1]  # Close the loop
                    ax.plot(angles, values, linewidth=2, label=scenario_name)
                    ax.fill(angles, values, alpha=0.1)
                
                # Set chart labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(radar_labels)
                
                # Add legend
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                
                plt.savefig(f"{results_dir}/scenario_comparison_radar.png", dpi=300, bbox_inches='tight')
                # plt.show()  # Comentado para evitar mostrar visualizaciones
                
            # Create combo visualization for overall algorithm performance summary
            plt.figure(figsize=(15, 8))
            plt.suptitle(f'Overall Algorithm Performance Summary: {algorithm_name}', fontsize=18)
            
            # Left plot: Bar chart showing overall metrics
            plt.subplot(1, 2, 1)
            metric_names = ['Success\nRate (%)', 'Path\nLength', 'Path\nSmoothness', 'SPL', 'SPS']
            metric_values = [overall_success, overall_path_length, overall_smoothness, overall_spl, overall_sps]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
            
            # Normalize metrics for better visualization
            norm_values = [
                overall_success,  # Keep percentage
                min(overall_path_length / 200, 1.0),  # Normalize path length
                overall_smoothness,  # Already 0-1
                overall_spl,        # Already 0-1
                overall_sps         # Already 0-1
            ]
            
            bars = plt.bar(metric_names, norm_values, color=colors, width=0.6)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, metric_values)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.ylim(0, 1.2)
            plt.title('Normalized Metrics Overview')
            
            # Right plot: Text summary with overall metrics
            plt.subplot(1, 2, 2)
            plt.axis('off')
            summary_text = f"""
            ALGORITHM: {algorithm_name.upper()}
            
            KEY PERFORMANCE INDICATORS:
            
            • Success Rate: {overall_success:.1f}%
            • Path Length: {overall_path_length:.2f}
            • Path Smoothness: {overall_smoothness:.4f}
            • SPL (Success weighted Path Length): {overall_spl:.4f}
            • SPS (Success weighted Path Smoothness): {overall_sps:.4f}
            
            Scenarios tested: {len(map_types)}
            Maps tested: {sum(len(all_scenario_results[s]) for s in map_types)}
            Episodes per map: {n_episodes}
            """
            plt.text(0, 0.5, summary_text, fontsize=12, va='center')
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/overall_performance_summary.png", dpi=300, bbox_inches='tight')
            # plt.show()  # Comentado para evitar mostrar visualizaciones
    
    print("\n=== All map tests completed ===")
    print(f"Results and visualizations saved to {results_dir}/")

