"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
from sample_players import improved_score
from sample_players import open_move_score
from sample_players import null_score


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    raise NotImplementedError


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.max_moves={}

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            pass

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        raise NotImplementedError

    def evaluate(self,game, location):

        result=0
        if location in game.get_legal_moves(game.inactive_player):
            result+=1
        new_game=game.forecast_move(location)
        moves = [i for i in new_game.get_legal_moves() if i in new_game.get_legal_moves(new_game.inactive_player)]
        print("moves "+str(moves))
        result -=len(moves)
        return result

        return len([i for i in game.get_legal_moves(game.active_player) if i not in game.get_legal_moves(game.inactive_player)])
        return null_score(game,game.active_player)
        return open_move_score(game,game.active_player)
        return improved_score(game,game.active_player)
        #return len(game.get_legal_moves())- len(game.get_legal_moves(game.inactive_player))

    def minValue(self,game,depth,maximizing_player=False):
      
        #print("===Min Value=====")
        #print(game.print_board())
        if depth==1:
            val,l = min([(self.score(game.forecast_move(m),game.active_player),m) for m in game.get_legal_moves()])  
                 
            #print("returning min for depth"+str(depth)+" location: "+str(l)+" is: "+str(val))
            return val,l
        # elif depth==1:
        #     return self.maxValue(game,depth,maximizing_player)
        v=float("Inf")
        legal_moves = game.get_legal_moves()
        #print("===Min Value=====")
       # print(game.print_board())
        #print("location: "+str(game.get_player_location(game.active_player))+" legal moves: "+str(legal_moves))
        location=None
        for i in game.get_legal_moves():
            new_game=game.forecast_move(i)
            val,l = self.maxValue(new_game,depth-1,maximizing_player=True)
            if v>val:
                v=val
                location=i
        #print("from max returning: "+str(v)+" with location: "+str(location))
        return v,location

    def maxValue(self,game,depth,maximizing_player=True):


       # print("===Max Value=====")
        #print(game.print_board())
       # print("location: "+str(game.get_player_location(game.active_player))+" legal moves: "+str(game.get_legal_moves()))
        if depth==1:
            val,l = max([(self.score(game.forecast_move(m),game.active_player),m) for m in game.get_legal_moves()])
            self.max_moves[depth]=(val,l)
            # self.last_max=val
            # self.last_max_location=l
            #print("returning max for depth: "+str(depth)+ " val "+str(val)+" location: "+str(l))
            
            return val,l

            #return len(game.get_legal_moves())+game.utility(game.active_player),game.get_player_location(game.active_player)
        v=-float("Inf")
        curr_score=v
        legal_moves = game.get_legal_moves()



        location=None
        store=[]

        for i in legal_moves:
 
            new_game=game.forecast_move(i)
            curr_score = self.score(new_game,game.active_player)
            if depth not in self.max_moves:
                self.max_moves[depth]=(curr_score,i)
                #print("depth: "+str(depth)+ "last max first time: "+str(curr_score)+" location: "+str(i))
            elif self.max_moves[depth][0]<curr_score:
                self.max_moves[depth]=(curr_score,i)
                #print("depth: "+str(depth)+ "last max: "+str(curr_score)+" location: "+str(i))          

            # if self.last_max is None:
            #     self.last_max=curr_score
            #     self.last_max_location=i
            #     print("depth: "+str(depth)+ "last max: "+str(self.last_max)+" location: "+str(self.last_max_location))
            # elif self.last_max<curr_score:
            #     self.last_max=curr_score
            #     self.last_max_location=i
            #     print("depth: "+str(depth)+ "last max: "+str(self.last_max)+" location: "+str(self.last_max_location))
            #     print("last max: "+str(self.last_max)+" location: "+str(self.last_max_location))
            
            val,l = self.minValue(new_game,depth-1,maximizing_player=False)
            
            if v<val:
                v=val
                location=i
            if depth==2:
                store.append((new_game,i))
            


            #print("i: "+str(i)+" v: "+str(v)+ "val: "+str(val)+" location "+str(location) + " l: "+str(l))
        
        #print(str(self.max_moves))
        if depth==2:
            #print(list(store))
            v,l = max([(self.score(new_game,new_game.inactive_player),i) for new_game,i in store])
            return v,l
                

        return v,location

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if depth==0 or not game.get_legal_moves():
            return self.score(game,game.active_player),game.get_player_location(game.active_player)
        best_move=None
        legal_moves = game.get_legal_moves()
        store=[]
        
        if maximizing_player:
            best_option=-float("Inf")

            for m in legal_moves:
                new_game = game.forecast_move(m)
                option,_=self.minimax(new_game,depth-1,False)
                
                if best_option < option:
                    best_option=option
                    best_move=m
                if depth==1:
                    store.append((new_game,m))
            if depth==1:
                best_option,best_move = max((self.score(game,game.inactive_player),m) for game,m in store)                
            return best_option,best_move
        
        else:
            best_option=float("Inf")
            for m in legal_moves:
                new_game = game.forecast_move(m)
                option,_=self.minimax(new_game,depth-1,True)
                if best_option > option:
                    best_option=option
                    best_move=m
            return best_option,best_move
        
                

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth==0 or not game.get_legal_moves():
            return self.score(game,game.active_player),game.get_player_location(game.active_player)
        best_move=None
        legal_moves = game.get_legal_moves()
        store=[]

        if maximizing_player:
            best_option=-float("Inf")
            for m in legal_moves:
                new_game = game.forecast_move(m)
                option,_=self.alphabeta(new_game,depth-1,alpha,beta,False)                
                
                if best_option < option:
                    best_option=option
                    best_move=m
                if depth==1:
                    store.append((new_game,m))
                alpha=max(alpha,best_option)
                if depth==1:
                    best_option,best_move = max((self.score(game,game.inactive_player),m) for game,m in store)  
                if beta<=alpha:
                    break
            return best_option,best_move
        else:
            best_option=float("Inf")
            for m in legal_moves:
                new_game = game.forecast_move(m)
                option,_=self.alphabeta(new_game,depth-1,alpha,beta,True)
                if best_option > option:
                    best_option=option
                    best_move=m
                beta=min(beta,best_option)
                if beta <= alpha:
                    break
            return best_option,best_move





