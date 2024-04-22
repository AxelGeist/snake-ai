# snake-ai

This repository contains the implementation of an AI-powered Snake game, using PyTorch for the reinforcement learning model. The AI uses a simple linear Q-learning network to decide the best moves based on the game state.

## Getting Started

### Prerequisites

- Anaconda
- Python 3.8+
- PyTorch
- Pygame
- Numpy


After setting up your Anaconda environment, install the necessary Python libraries with `Anaconda Prompt`:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

```bash
pip install torch pygame numpy
```

### Structure

- `game.py`: Contains the game environment including the snake mechanics and rules.
- `model.py`: Defines the Linear Q-Network and the Q-learning trainer.
- `agent.py`: Contains the logic for the AI agent, including state management and decision-making processes.
- `helper.py`: Utility script for plotting the performance metrics.

### Running the Game

1. **Activate your virtual environment**:
   If you haven't created a virtual environment for this project, you can create one using:

   ```
   conda create --name pygame_env python=3.8
   ```

    Activate the environment with:

    ```
    conda activate pygame_env
    ```

2. **Then run the game with the following script**:

    ```bash
    python agent.py
    ```

This will open a Pygame window and start the game. The AI will learn how to play the game over time by adjusting its strategy based on the reward system.

### How It Works

The AI agent uses a reinforcement learning model where it decides between exploring new moves or exploiting known strategies based on an epsilon-greedy approach. The state of the game is represented as a vector of binary values, which indicate the immediate risk of collision, the current direction of movement, and the position of the food relative to the snake's head.

### Code Highlights

The AI's decision-making process is centered around the calculation of the game state and the subsequent prediction using a trained model. Here’s a quick look at the core functionality:

```python
def get_action(self, state):
    # Exploration vs Exploitation
    self.epsilon = max(80 - self.n_games, 0)
    final_move = [0,0,0]
    if random.randint(0, 200) < self.epsilon:
        move = random.randint(0, 2)
        final_move[move] = 1
    else:
        state0 = torch.tensor(np.array(state), dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

    return final_move
```

The network is trained after each game round to improve its predictions:

```python
def train_long_memory(self):
    if len(self.memory) > BATCH_SIZE:
        mini_sample = random.sample(self.memory, BATCH_SIZE)
    else:
        mini_sample = self.memory

    states, actions, rewards, next_states, dones = zip(*mini_sample)
    self.trainer.train_step(states, actions, rewards, next_states, dones)
```

### Model Saving

The model automatically saves its best performing weights to a file, allowing the AI to resume its progress:

```python
def save(self, file_name='model.pth'):
    model_folder_path = './model'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    file_name = os.path.join(model_folder_path, file_name)
    torch.save(self.state_dict(), file_name)
```

## Current record

### Score: 103 after 2274 games played

![image](https://github.com/AxelGeist/snake-ai/assets/73136957/670c8f2b-4e91-4da0-bddb-b6bf0589b22f)

### Score: 70 after 152 games played

![image](https://github.com/AxelGeist/snake-ai/assets/73136957/de2742e7-a36e-4ea0-b161-9643c54f4f1d)

