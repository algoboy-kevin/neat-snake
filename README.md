# neat-snake

This Python repository contains a testing environment and implementation of the NEAT algorithm for the classic game of Snake.

The Snake game is played on a 10x10 map. Each time the snake eats an apple, its length increases and the fitness score is incremented by 1. The goal is to achieve a fitness score of 20.

## Installation
To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage
There are several options available for running the Snake game:

- **Test Environment**: To test the environment, execute the following command:
```
python main.py test
```

- **Train Snake**: To train the snake using the NEAT algorithm, use the following command:
```
python main.py train
```

- **Save and Create Winner Model**: To save and create a winner model after training, run the following command:
```
python main.py save
```

- **Run Trained Model**: To run a trained model, execute the following command:
```
python main.py run
```

- **Run Master Model**: To run the master model, execute the following command:
```
python main.py run_master
```

Note: The last two commands are the same, but provided separately for clarity.


## Contributors
- Snake env logic and network render reference. [Daniel Chang](https://github.com/danielchang2002)

Feel free to contribute to this project by submitting pull requests or suggesting improvements. Happy coding!