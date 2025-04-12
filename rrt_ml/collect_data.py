import gymnasium as gym
import numpy as np
from indoor_robot_env import IndoorRobotEnv # Đảm bảo import đúng env
from indoor_robot_controller import IndoorRobotController # Đảm bảo import đúng controller
import matplotlib.pyplot as plt
import os.path as path

def run_simulation_and_collect(env, controller, num_episodes=10, render=False):
    """Runs simulation episodes and collects data."""

    controller.start_data_collection() # Bắt đầu thu thập và xóa dữ liệu cũ

    for episode in range(num_episodes):
        print(f"\n--- Starting Episode {episode + 1}/{num_episodes} ---")
        observation, info = env.reset()
        controller.reset() # Reset controller state for new episode
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        while not terminated and not truncated:
            action, controller_info = controller.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if render:
                env.render(mode='human', controller_info=controller_info)
                # Optional: Add status text to render
                if controller_info and 'status' in controller_info:
                     plt.gca().set_title(f"Indoor Robot Sim - Step: {step_count} Status: {controller_info['status']}", fontsize=10)


            if terminated or truncated:
                status = info.get('status', 'unknown')
                print(f"Episode {episode + 1} finished after {step_count} steps. Status: {status}. Total Reward: {total_reward:.2f}")

    controller.stop_data_collection() # Dừng thu thập
    env.close()

# --- Main Execution ---
if __name__ == "__main__":
    # Khởi tạo môi trường và controller
    # render_mode='human' để xem, 'rgb_array' nếu không cần xem hoặc chạy trên server
    env = IndoorRobotEnv(render_mode='rgb_array', max_steps=500)
    controller = IndoorRobotController(env)

    num_collection_episodes = 50 # Số episode để chạy thu thập dữ liệu

    # Chạy mô phỏng và thu thập
    run_simulation_and_collect(env, controller, num_episodes=num_collection_episodes, render=True) # Đặt render=True nếu muốn xem

    # Lấy dữ liệu đã thu thập
    collected_data = controller.get_collected_data()

    # Lưu dữ liệu
    if collected_data:
        controller.save_collected_data(path.join(path.dirname(path.abspath(__file__)), 'data', 'avoidance_dataset.pkl'))

        # (Optional) Kiểm tra nhanh dữ liệu
        print(f"\n--- Sample Data Check ---")
        print(f"Number of samples: {len(collected_data)}")
        if len(collected_data) > 0:
            sample_state, sample_action = collected_data[0]
            print(f"Sample State (observation) shape: {sample_state.shape}")
            print(f"Sample Action shape: {sample_action.shape}")
            print(f"Sample State (first 10 elements): {sample_state[:10]}")
            print(f"Sample Action: {sample_action}")
    else:
        print("No data was collected.")