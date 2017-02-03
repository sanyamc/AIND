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

#TODO: add docstrings
# about 90% on 3 different tries
def heuristic_first(game,player):
    result=0
    if game.get_player_location(player) in game.get_legal_moves(game.inactive_player):
        result+=1
    #new_game=game.forecast_move(location)
    moves = [i for i in game.get_legal_moves() if i in game.get_legal_moves(game.inactive_player)]
    #print("moves "+str(moves))
    result -=len(moves)
    return float(result)



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
    return heuristic_first(game, player)

    #return null_score(game,player)

    


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


    def is_symmetric_move(game, move):
        inactive=game.get_player_location(game.inactive_player)
        r,c=inactive[0]-i[0],inactive[1]-i[1]
        if game.move_is_legal((r,c)):
            return True
        return False



            
        


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

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        #print("legal_moves: "+str(legal_moves))
        if len(legal_moves)==0:
            return (-1,-1)
        best_move=(-1,-1)




                

                
            # pick a move which eliminates other player's options
            # for move in game.get_legal_moves():
            #     if move in game.get_legal_moves(game.inactive_player):
            #         new_game = game.forecast_move(move)
            #         if new_game.get_legal_moves(game.inactive_player)>= new_game.get_legal_moves():
            #             return move

            #print("first move was mine")


            #print("first: "+str(len(game.get_blank_spaces())))

        # start with center
       # if    return (game.height//2,game.width//2)
        
        

        



        try:

            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if len(game.get_blank_spaces())>=38:
                val,move=max([(improved_score(game.forecast_move(m),game.active_player),m) for m in game.get_legal_moves()])
                print(val)
                #print("returning move")
                return move

            #     #print("moves left: "+str(len(game.get_blank_spaces())))
            #     print(game.print_board())
            #     print("current loc: "+str(game.get_player_location(game.active_player)))
            #     center=(3,3)                
            #     if center in game.get_legal_moves():
            #         print("returning center")                    
            #         return center
            #     opening_moves=[(2,2),(2,3),(2,4),(3,2),(3,4),(4,2),(4,3),(4,4)]
            #     for move in game.get_legal_moves():
            #         if move in opening_moves and move in game.get_legal_moves(game.inactive_player):
                        
                        
            #             print("returning move: "+str(move))
            #             return move

            if self.iterative:            
                i=0
                while(i!=self.search_depth):                    
                    if self.method=="minimax":
                        val,best_move=self.minimax(game,i,True)
                    else:
                        val,best_move=self.alphabeta(game,i)
                    i+=1
            else:
                
                if self.method=="minimax":
                    val,best_move=self.minimax(game,self.search_depth,True)
                else:
                    val,best_move=self.alphabeta(game,self.search_depth)     

        except Timeout:
            # Handle any actions required at timeout, if necessary
            #return best_move
            pass

        # Return the best move from the last completed search iteration
        return best_move

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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth==0 or len(game.get_legal_moves())==0:
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

        if depth==0 or len(game.get_legal_moves())==0:
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





