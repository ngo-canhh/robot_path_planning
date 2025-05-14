from indoor_robot_env import IndoorRobotEnv
from components.obstacle import StaticObstacle, DynamicObstacle
from components.shape import Circle, Rectangle, Triangle, Polygon
from indoor_robot_controller_raytracing import IndoorRobotController
import matplotlib.pyplot as plt
import numpy as np
import math

BASE_SEED = 88

if __name__ == "__main__":
    print("Running OOP Obstacle Simulation...")
    # # Use the modified environment
    config_path = 'convert_map/config_map/map2.yaml'
    # config_path = 'convert_map/config_map/Sydney_2_256.yaml'
    start = (10, 10)
    goal = (80, 10)
    env = IndoorRobotEnv(config_path=config_path, start=start, goal=goal, robot_radius=1, sensor_range=15, render_mode='human')
    # env = IndoorRobotEnv(width=500, height=500, sensor_range=150, render_mode='human', obs_chance_dynamic=0.7)

    # Add dynamic obstacles
    env.add_obstacles([
        # DynamicObstacle(22, 40, Circle(5), 2, (-1, 1), (10, 30, 40, 50)),
        DynamicObstacle(50, 53, Circle(2.5), 2, (-1, 1), (40, 40, 60, 60)),
        DynamicObstacle(95, 15, Circle(2.5), 2, (-1, 1), (80, 30, 100, 40)),
        DynamicObstacle(95, 15, Circle(2.5), 2, (-1, 1), (60, 30, 80, 40)),
        DynamicObstacle(95, 15, Circle(2.5), 2, (-1, 1), (20, 40, 40, 60)),
    ])

    # Use the modified controller
    controller = IndoorRobotController(env)

    max_episodes = 10
    episode_rewards = []
    episode_steps = []
    episode_status = []

    for episode in range(max_episodes):
        print(f"\n--- Starting Episode {episode+1} ---")
        observation, info = env.reset(seed=BASE_SEED + episode) # Ensure variety
        controller.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        controller_info = None

        if env.render_mode == 'human':
            env.render(controller_info=controller_info)
            # plt.pause(0.5) # Optional pause

        while not terminated and not truncated:
            step_count += 1
            action, controller_info = controller.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            if env.render_mode == 'human':
                env.render(controller_info=controller_info)
            elif env.render_mode == 'rgb_array':
                 img = env.render(controller_info=controller_info)
                 # Process img

            total_reward += reward

            if step_count > env.max_steps + 50: # Extra safety break
                print("Warning: Exceeded max steps significantly, breaking loop.")
                truncated = True


        status = info.get('status', 'unknown')
        print(f"Episode {episode+1} finished after {step_count} steps.")
        print(f"Status: {status}")
        print(f"Total reward: {total_reward:.2f}")
        episode_rewards.append(total_reward)
        episode_steps.append(step_count)
        episode_status.append(status)

        if env.render_mode == 'human' and (terminated or truncated):
             print("Episode end. Pausing...")
             try: plt.pause(1.0)
             except Exception as e: print(f"Error during pause: {e}")

    env.close()
    print("\n--- Simulation Summary (OOP Obstacles) ---")
    print(f"Ran {max_episodes} episodes.")
    print(f"Final Statuses: {episode_status}")
    print(f"Episode Steps: {episode_steps}")
    print(f"Episode Rewards: {[f'{r:.2f}' for r in episode_rewards]}")
    if episode_rewards:
         print(f"Average Steps: {np.mean(episode_steps):.1f}")
         print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print("---------------------------------------------")

