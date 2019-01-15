# Welcome to TDComparator

This library has been made to enable you to run and compare easily Temporal-Difference algorithm on Markov Reward Process. It contains many classes to build quickly and easily your model.

Three algorithms have been implemented, the On-TD(0), the Off-TD(0) and the [emphatic TD of Sutton & al. (2015)](https://arxiv.org/abs/1507.01569). This library follows the works of Sutton and thus implements the different examples found in their paper.

The algorithm and formula used in the library are all from the paper of [Sutton & al. (2015)](https://arxiv.org/abs/1507.01569) The paper is freely available [here](https://arxiv.org/abs/1507.01569).


## Requirements
The library has very few requirements :
 - Python 3
 - Numpy
 - Matplotlib

## Installation
To use the class of the library, you just need to import its main folder to your Python. You can do it like that :
```python
import sys
sys.path.insert(0, "library/") # Path of the library folder on your computer
```

## Documentation
To understand the library, you can look at the example in the folder "examples".  This is a list of small tutorials :

 1. [The basics](https://github.com/Nicotous1/EmpathicTD/blob/master/examples/1%20-%20The%20basics.ipynb) : create a two states model and run the emphatic TD.
 2. [Comparing algorithms](https://github.com/Nicotous1/EmpathicTD/blob/master/examples/2%20-%20Comparing%20algorithms.ipynb) : create a five states model and compare the off-TD(0) and the emphaticTD(0)
 3. [Tuning hyper-parameters](https://github.com/Nicotous1/EmpathicTD/blob/master/examples/3%20-%20Tuning%20hyper-parameters.ipynb) : Quick optimization of alpha and lambda for the emphatic-TD
 4. [2D grid](https://github.com/Nicotous1/EmpathicTD/blob/master/examples/4%20-%202D%20grid.ipynb) : Create a 5x5 grid and run the off-TD(0) and the emphatic-TD(0)

## Files structure
The library contains 4 files, I will briefly describe what they contain :

 - [TD.py](https://github.com/Nicotous1/EmpathicTD/blob/master/library/TD.py) -> contains all the TD algorithms (inherited from AbstractTD)
	 - Off-TD(0)
	 - Emphatic-TD from Sutton
 - [policies.py](https://github.com/Nicotous1/EmpathicTD/blob/master/library/policies.py) -> differents policies (inherited from Policy)
	 -  RightOrLeft : move right or left defined by the probability of right or left
	 - GridRandomWalk : a random walk defined by the probabilities of up, down, left or right.
 - [models.py](https://github.com/Nicotous1/EmpathicTD/blob/master/library/models.py) -> contains the model to store your parameters
	 - Model : the basic class to store your parameter.
	 - Grid : A class to quickly create a grid model
 - [utils.py](https://github.com/Nicotous1/EmpathicTD/blob/master/library/utils.py) -> useful tools to analyse and paralelize the computation with numpy
	 - comparatorTD : the tool to compute and compare the TD
