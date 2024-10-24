# Lunar Lander with Deep Q-Learning

This project implements a Deep Q-Network (DQN) agent to solve OpenAI Gym's Lunar Lander environment. The agent learns to safely land a lunar module on a landing pad using reinforcement learning techniques.

## Environment Description

In this environment, the agent needs to control a lunar lander and safely land it on a designated landing pad. The landing pad is marked by two flag poles and is centered at coordinates (0,0). The lander starts at the top center of the environment with a random initial force and has infinite fuel.

### State Space
The agent receives an 8-dimensional state observation:
- Coordinates (x, y)
- Linear velocities
- Angle
- Angular velocity
- Boolean flags for left and right leg ground contact

### Action Space
The agent can take 4 discrete actions:
- 0: Do nothing
- 1: Fire right engine
- 2: Fire main engine
- 3: Fire left engine

### Rewards
The reward structure includes:
- Proximity rewards based on distance to landing pad
- Velocity-based rewards (slower movement is better)
- Penalties for tilted angles
- +10 points for each leg contact with ground
- -0.03 points per frame for side engine usage
- -0.3 points per frame for main engine usage
- Terminal rewards: +100 for safe landing, -100 for crash

## Project Structure

- `lunar_lander.py`: Main implementation file
- `utils.py`: Helper functions for the project
- Requirements:
  - numpy
  - tensorflow
  - gym (version 0.24.0)
  - PIL
  - pyvirtualdisplay

## Technical Implementation

### Deep Q-Network Architecture
- Input layer: State size (8 dimensions)
- Hidden layers: 2 dense layers with 64 units each (ReLU activation)
- Output layer: 4 units (linear activation) corresponding to actions

### Key Features
1. **Experience Replay**: Stores agent experiences in a memory buffer to break correlations between consecutive samples
2. **Target Network**: Separate network for stable Q-value targets
3. **ε-greedy Exploration**: Balanced exploration-exploitation strategy
4. **Custom Video Generator For Each Landing**: After training U can run Below Cells To get A Video Generated for Your Custom Lunar Lander Video Where Each Video will have Different Landing Position.

![Demo GIF](https://github.com/sohailshk/Lunar-Lander-AI-Safe-Descent/blob/a6c63fd07abff321aef3d9f8cdba22ec36a31aa7/Lunar_Lander_Files/download-ezgif.com-video-to-gif-converter.gif)


   

## Training Parameters

- Episodes: 2000
- Max timesteps per episode: 1000
- Memory buffer size: Defined in hyperparameters
- Batch size: 64
- Learning rate (α): Defined in hyperparameters
- Epsilon decay: Progressive reduction from 1.0 to 0.01

## Success Criteria

The environment is considered solved when the agent achieves an average score of 200 points over 100 consecutive episodes.

## Usage

```python
# Initialize environment
env = gym.make('LunarLander-v2')

# Train agent
trained_agent = train_dqn_agent(env)

# Watch trained agent
create_video(trained_agent, env, "lunar_landing.mp4")
```

## Dependencies Installation

```bash
pip install numpy tensorflow gym==0.24.0 pillow pyvirtualdisplay
```

## Notes

- The agent uses random initial forces, so each landing attempt will be different
- Training time can vary based on hardware capabilities
- The implementation includes visualization capabilities for monitoring training progress

## License

[Include your license information here]
