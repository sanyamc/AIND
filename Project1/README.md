# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: To solve nake twins i made sure that when a box matches with another box value in the unit AND it is of length 2 then we could eliminate those values from the peers.
   So the constraint was that no other box in that unit can have either of twin values. We could extend the solution to naked triplets or naked quadruplets.
   I think naked twins is a constraint we use to solve the sudoku puzzle. Like eliminate and only choice, this reduces search space.
   Reason we use constraint propagation is to reduce the search space which without constraint propogation would be huge for sudoku.
   For e.g. every box will have one of 9 choices and there are 81 boxes so search space would be around 9^81 combinations. 
   Trying every value in every box would be time consuming on a computer. Hence we use constraint propogation techniques with search to solve sudoku puzzle.



# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: To solve the diagnol sudoku; i changed the unitlist and peers to add additional boxes to both of them and then used eliminate, only choice and 
   optionally naked twins(uncomment in reduce_puzzle) to reduce the puzzle's problem space and then brute forced it with a search algorithm.
   On every step we reduce the puzzle using constraint propogation techniques. 
   As mentioned above constraint propogation reduces the problem space but depending on problem it might not solve it completely; Hence we need to search and reduce the space 
   until we reach the solution.

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solutions.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in function.py

### Data

The data consists of a text file of diagonal sudokus for you to solve.