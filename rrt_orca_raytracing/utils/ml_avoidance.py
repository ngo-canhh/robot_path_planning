import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
class MLAvoidance(ABC):
    """
    Class for ML avoidance.
    """

    def __init__(self, model_path: str):
        """
        Initialize the MLAvoidance class.

        Args:
            model_path (str): Path to the ML model.
        """
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def load_model(self):
        """
        Load the ML model.
        """
        # Load the model from the specified path
        pass

    @abstractmethod
    def predict(self, data):
        """
        Predict avoidance actions using the ML model.

        Args:
            data: Input data for prediction.

        Returns:
            Predicted actions.
        """
        # Use the loaded model to predict actions
        pass


# Torch model for robot avoidance
class RobotAvoidanceNetwork(nn.Module):
    def __init__(self, obs_robot_state_size, obs_obstacle_data_size):
        self.obs_robot_state_size = obs_robot_state_size
        self.obs_obstacle_data_size = obs_obstacle_data_size
        super(RobotAvoidanceNetwork, self).__init__()

        self.state_mlp = nn.Sequential(
            nn.Linear(self.obs_robot_state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.attention_obs_mlp = nn.Sequential(
            nn.Linear(self.obs_obstacle_data_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.obs_mlp = nn.Sequential(
            nn.Linear(self.obs_obstacle_data_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.final_mlp_out = nn.Sequential(
            nn.Linear(16 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, observations): # N, obs_robot_state_size + num_obstacles * obs_obstacle_data_size
        # print(observations.shape)
        num_obstacles = int((observations.shape[1] - self.obs_robot_state_size) // self.obs_obstacle_data_size)
        robot_state = observations[:, :self.obs_robot_state_size]
        obstacle_data = observations[:, self.obs_robot_state_size:]
        obstacle_data = obstacle_data.view(-1, num_obstacles, self.obs_obstacle_data_size) # N, num_obstacles, obs_obstacle_data_size

        # Robot state
        robot_state = self.state_mlp(robot_state) # N, 16

        # Obstacle data
        obstacle_attention = F.softmax(torch.sum(self.attention_obs_mlp(obstacle_data), dim=2, keepdim=True), dim=1)
        obstacle_data = torch.mul(obstacle_data, obstacle_attention) # N, num_obstacles, obs_obstacle_data_size
        obstacle_data = self.obs_mlp(obstacle_data) # N, num_obstacles, 16
        obstacle_data = torch.sum(obstacle_data, dim=1) # N, 16
        
        # Concatenate robot state and obstacle data
        x = torch.cat((robot_state, obstacle_data), dim=1)
        x = self.final_mlp_out(x)
        return x

# model to use in the controller
class SimpleMLPAvoidance(MLAvoidance):
    """
    Class for ML avoidance using a feedforward neural network.
    """
    MODEL_PATH = "/Users/ngocanhh/Documents/Study/tinhToanTienHoa/Lab_Robot/rrt_ml/data/robot_avoidance_model.pth" 
    def __init__(self, obs_robot_state_size, obs_obstacle_data_size):
        """
        Initialize the MLPAvoidance class.

        Args:
            model_path (str): Path to the ML model.
        """
        super().__init__(model_path=self.MODEL_PATH)
        self.obs_robot_state_size = obs_robot_state_size
        self.obs_obstacle_data_size = obs_obstacle_data_size
        self.load_model()

    def load_model(self):
        """
        Load the ML model.
        """
        # Load the model from the specified path
        self.model = RobotAvoidanceNetwork(self.obs_robot_state_size, self.obs_obstacle_data_size)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, data):
        """
        Predict avoidance actions using the ML model.

        Args:
            data: Input data for prediction.

        Returns:
            Predicted actions.
        """
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            predictions = self.model(data_tensor)
            predictions = predictions.squeeze()
            return predictions.numpy()