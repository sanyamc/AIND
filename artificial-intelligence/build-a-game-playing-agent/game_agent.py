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



# try to order moves to the front in case they were found in shallow depth; this will enable chances of better pruning
def get_move_order(game,game_hash,tt):
    moves = game.get_legal_moves()
    if game_hash in tt:
        val = tt[game_hash]
        if val in moves:
            moves.pop(moves.index(val))
            moves.insert(0,val)
    return moves


def hash(game):
    aloc = str(game.__last_player_move__[game.__active_player__])
    iloc = str(game.__last_player_move__[game.__inactive_player__])
    state = str([bool(x) for x in sum(game.__board_state__, [])])
    return aloc + iloc + state


# heuristic which penalizes moves which are in legal moves of opponent player; especially if that move is
# what opponent most likely to take
# with improvement over improved_score
def heuristic_penalty(game,player):
        
    result=heuristic_priority(game, player)
    
    val=heuristic_penalty_helper(game,player)
    result+=val
    return float(result)

def heuristic_final(game, player):
    if len(game.get_blank_spaces())>30:
        return heuristic_priority(game,player)
    result = improved_score(game, player)
    result+=heuristic_penalty_helper(game,player)
    return result   


def heuristic_penalty_helper(game,player):
    oppn_moves = game.get_legal_moves(game.get_opponent(player))
    moves = [i for i in game.get_legal_moves(player) if i in oppn_moves ]
    weight=0
    eights=[(3,3),(3,2),(3,4),(2,3),(2,2),(2,4),(4,3),(4,2),(4,4)]
    sixes=[(1,2),(1,4),(5,2),(5,4),(1,3),(5,3),(2,1),(3,1),(4,1),(2,5),(3,5),(4,5)]
    fours=[(1,1),(1,5),(5,1),(5,5),(0,2),(0,3),(0,4),(2,0),(3,0),(4,0),(2,6),(3,6),(4,6)]
    rims_and_corners=[(0,1),(0,5),(6,1),(6,5),(5,0),(5,6),(1,0),(1,6),(0,0),(0,6),(6,0),(6,6)]
    eight_moves=[4 for i in moves if i in eights]
    six_moves = [3 for i in moves if i in sixes]
    four_moves = [2 for i in moves if i in fours]

    remain = [1 for i in moves if i in rims_and_corners]
    weight+=sum(eight_moves)
    weight+=sum(six_moves)
    weight+=sum(four_moves)
    weight+=sum(remain)
    return -float(weight)

def heuristic_priority(game,player):
    eights=[(3,3),(3,2),(3,4),(2,3),(2,2),(2,4),(4,3),(4,2),(4,4)]
    sixes=[(1,2),(1,4),(5,2),(5,4),(1,3),(5,3),(2,1),(3,1),(4,1),(2,5),(3,5),(4,5)]
    fours=[(1,1),(1,5),(5,1),(5,5),(0,2),(0,3),(0,4),(2,0),(3,0),(4,0),(2,6),(3,6),(4,6)]
    rims_and_corners=[(0,1),(0,5),(6,1),(6,5),(5,0),(5,6),(1,0),(1,6),(0,0),(0,6),(6,0),(6,6)]

    #board={(3,3):50,(3,2):}

    all_moves = eights + sixes + fours + rims_and_corners
    location = game.get_player_location(player)

    if location in eights:
        val=50
    elif location in sixes:
        val=20
    elif location in fours:
        val=0
    else:
        val=-50
    
    return float(val)




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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    return heuristic_final(game, player)

    


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
        self.tt={} # got the idea from https://en.wikipedia.org/wiki/Transposition_table   
     


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
        if len(legal_moves)==0:
            return (-1,-1)
        best_move=legal_moves[0]

        
        self.tt={}
        game_hash = hash(game)        

        try:         

            if self.iterative:            
                i=0
                sentinel = float("Inf")
                if self.search_depth>=0:
                    sentinel=self.search_depth
                while(i<=sentinel):                    
                    if self.method=="minimax":
                        val,best_move=self.minimax(game,i,True)
                    else:
                        val,best_move=self.alphabeta(game,i,float("-inf"),float("inf"),True,game_hash)
                        self.tt[game_hash]=best_move
                    i+=1
            else:                
                if self.method=="minimax":
                    val,best_move=self.minimax(game,self.search_depth,True)
                else:
                    val,best_move=self.alphabeta(game,self.search_depth,float("-inf"),float("inf"),True, game_hash)     

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass
        finally:           
            return best_move
        
  

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

        if depth==0 and maximizing_player==False: # triggered by max move
            return self.score(game,game.inactive_player),game.get_player_location(game.inactive_player)
        elif depth==0 and maximizing_player==True: # triggered by min move
            return self.score(game,game.active_player),game.get_player_location(game.active_player)

        
        legal_moves = game.get_legal_moves()
        best_move=None
        
        if maximizing_player:
            best_option=-float("Inf")

            for m in legal_moves:
                new_game = game.forecast_move(m)
                option,_=self.minimax(new_game,depth-1,False)
                
                if best_option < option:
                    best_option=option
                    best_move=m            
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
        
        if best_move==None and len(legal_moves)>0:
            # no matter what we select; agent either wins for sure or looses for sure
            best_move=legal_moves[0]
        return best_option,best_move
        
                
    

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True, game_hash=None):
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
       
        if depth==0 and maximizing_player==False:
            return self.score(game,game.inactive_player),game.get_player_location(game.inactive_player)
        elif depth==0 and maximizing_player==True:
            return self.score(game,game.active_player),game.get_player_location(game.active_player)
        
        legal_moves = get_move_order(game,game_hash,self.tt)    
        best_move=None

        if maximizing_player:
            best_option=-float("Inf")
            for m in legal_moves:
                new_game = game.forecast_move(m)
                option,_=self.alphabeta(new_game,depth-1,alpha,beta,False,game_hash)     
                
                if best_option < option:
                    best_option=option
                    best_move=m                    
                alpha=max(alpha,best_option)  
                
                if beta<=alpha:
                    break


        else:
            best_option=float("Inf")
            for m in legal_moves:
                new_game = game.forecast_move(m)
                option,_=self.alphabeta(new_game,depth-1,alpha,beta,True,game_hash)
                if best_option > option:
                    best_option=option
                    best_move=m
             
                beta=min(beta,best_option)
                if beta <= alpha:
                    break
        if best_move==None and len(legal_moves)>0:
            # no matter what we select; agent either wins for sure or looses for sure
            best_move=legal_moves[0]

        return best_option,best_move



