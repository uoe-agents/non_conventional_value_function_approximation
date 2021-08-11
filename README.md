# Non-conventional Function Approximation methods in Reinforcement Learning

This project evaluates and compares different approaches to function approximation in the Reinforcement Learning.

## Getting started

Create and activate virtual environment:
```
python3 -m venv [name_of_venv]
source [name_of_venv]/bin/activate
```

Clone repository:
```
git clone https://github.com/atsiakkas/rl_vfa.git
```

Install requirements:
```
cd rl_vfa
pip install -e .
```

## Project

https://github.com/atsiakkas/rl_vfa


## Contents

**agents**: Defines the classes of the RL agents: DQNAgent, LinearAgent, FQIAgent, OnlineGaussianProcessAgent

**custom_envs**: Defines the classes of the custom environments: SimpleGridworld, Windygridworld

**function_approximators**: Defines the classes of the function approximation models and of the replay buffer: ParametricModel, NeuralNetwork, LinearModel, NonParametricModel, DecisionTree, RandomForest, ExtraTrees, GradientBoostingTrees, SupportVectorRegressor, KNeighboursRegressor, GaussianProcess, eGaussianProcess, OnlineGaussianProcess

**plots**: Scripts (jupyter notebooks) for producing the plots used in the report and saved plots.

**results**: Saved output of runs (csv files).

**train**: Scripts (jupyter notebooks and .py files) for training and evaluation.

**utils**: Defines the training and plotting utility functions.