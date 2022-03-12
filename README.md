# Non-conventional Value Function Approximation methods in Reinforcement Learning

This project evaluates and compares different value function approximation methods in Reinforcement Learning using a range of parametric and non-parametric function approximation models. The parametric models (Neural Network and Linear Model) were implemented under the Deep Q-Network architecture [1] using the PyTorch framework [2] for their training. The non-parametric models (Decision Tree, Random Forest, Support Vector Regression, k-Nearest Neighbours, Gaussian Process) were implemented under the Fitted-Q Iteration architecture [3] and were defined through the Scikit-learn library [4]. Finally, the Online Gaussian Process model was implemented from scratch following the work of [5].

### Function approximation models evaluated
  1. Neural Network
  2. Linear Model
  3. Decision Tree
  4. Random Forest
  5. Support Vector Regression
  6. K-Nearest Neighbours Regression
  7. Gaussian Processes
  8. Online Gaussian Processes

### Environments
  1. SimpleGridworld
  2. WindyGridworld
  3. CartPole
  4. LunarLander

### Evaluation Criteria
  1. Performance
  2. Reliability
  3. Sample efficiency
  4. Training time
  5. Interpretability
<br/>

## Getting started

Create and activate virtual environment:
```
python3 -m venv [name_of_venv]
source [name_of_venv]/bin/activate
```

Clone repository:
```
git clone https://github.com/atsiakkas/non_conventional_value_function_approximation.git
```

Install requirements:
```
cd non_conventional_value_function_approximation
pip install -e .
```
<br/>



## Project

https://github.com/uoe-agents/non_conventional_value_function_approximation<br/>
<br/>


## Contents

**agents**: Defines the classes of the RL agents: 
 - DQNAgent
 - LinearAgent
 - FQIAgent
 - OnlineGaussianProcessAgent

**custom_envs**: Defines the classes of the custom environments:
 - SimpleGridworld
 - Windygridworld

**function_approximators**: Defines the classes of the function approximation models:
 - ParametricModel
 - NeuralNetwork
 - LinearModel
 - NonParametricModel
 - DecisionTree
 - RandomForest
 - ExtraTrees
 - GradientBoostingTrees
 - SupportVectorRegressor
 - KNeighboursRegressor
 - GaussianProcess
 - eGaussianProcess
 - OnlineGaussianProcess

**plots**: Scripts (jupyter notebooks) for producing the plots used in the report and saved plots.

**results**: Saved output of runs (csv files).

**train**: Scripts (jupyter notebooks and .py files) for training and evaluation.

**utils**: Defines the training and plotting utility functions.<br/>
<br/>


## References

[1] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. nature, 518(7540), pp.529-533.

[2] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L. and Desmaison, A., 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

[3] Ernst, D., Geurts, P. and Wehenkel, L., 2005. Tree-based batch mode reinforcement learning. Journal of Machine Learning Research, 6, pp.503-556.

[4] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V. and Vanderplas, J., 2011. Scikit-learn: Machine learning in Python. the Journal of machine Learning research, 12, pp.2825-2830.

[5] Csató, L. and Opper, M., 2002. Sparse on-line Gaussian processes. Neural computation, 14(3), pp.641-668.
