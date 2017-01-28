rows = 'ABCDEFGHI'
cols = '123456789'

def cross(a, b):
    return [s+t for s in a for t in b]

boxes = cross(rows, cols)

row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
first_diagnol_unit = [rows[i]+cols[i] for i in range(len(rows))]
second_diagnol_unit = [rows[i]+cols[len(rows)-i-1] for i in range(len(rows))]
unitlist = row_units + column_units + square_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)

# with the introduction of diagnol set; some boxes will have additional peers
# creating a copy and modifying new variable as other test expects normal sudoku for testing naked twins
peersDiagnol = peers.copy() 
for i in first_diagnol_unit:
    peersDiagnol[i]=peersDiagnol[i].union(set(first_diagnol_unit)-set([i]))
for i in second_diagnol_unit:
    peersDiagnol[i]=peersDiagnol[i].union(set(second_diagnol_unit)-set([i]))    

# similarly include diagnol as another unit
diagnol_unit_list = unitlist.copy() + [first_diagnol_unit] + [second_diagnol_unit]  


def display(values):
    """
    Display the values as a 2-D grid.
    Input: The sudoku in dictionary form
    Output: None
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Input: A grid in string form.
    Output: A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    chars = []
    digits = '123456789'
    for c in grid:
        if c in digits:
            chars.append(c)
        if c == '.':
            chars.append(digits)
    assert len(chars) == 81
    return dict(zip(boxes, chars))
    
def eliminate(values):
    """
    Go through all the boxes, and whenever there is a box with a value, eliminate this value from the values of all its peers.
    Input: A sudoku in dictionary form.
    Output: The resulting sudoku in dictionary form.
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peersDiagnol[box]:
            values[peer] = values[peer].replace(digit,'')
    return values

def only_choice(values):
    """
    Go through all the units, and whenever there is a unit with a value that only fits in one box, assign the value to this box.
    Input: A sudoku in dictionary form.
    Output: The resulting sudoku in dictionary form.
    """    
    for unit in diagnol_unit_list:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit

    return values


assignments = []

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values, unitlist=unitlist, peers=peers, reduce_after_unit=False):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    for unit in unitlist:
         # goes through all the units

        for box in unit:
            naked_cases = [peer for peer in peers[box] if values[box]==values[peer] and len(values[peer])==2] # we colud probably extend it to find naked triplets or quadruplets
            if len(naked_cases)==len(values[box])==2: 
                for value in values[box]:
                    for peer in peers[box]:
                        if peer in naked_cases:
                            continue
                        else:
                            values[peer]=''.join(values[peer].split(value))
        if reduce_after_unit: # used when using nake twins on diagnol sudoku
            values=eliminate(values)
       
    return values

def reduce_puzzle(values):
    """
    Iterate eliminate() and only_choice(). If at some point, there is a box with no available values, return False.
    If the sudoku is solved, return the sudoku.
    If after an iteration of both functions, the sudoku remains the same, return the sudoku.
    Input: A sudoku in dictionary form.
    Output: The resulting sudoku in dictionary form.
    """
    stalled = False
    
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        values = naked_twins(values, diagnol_unit_list, peersDiagnol, True) # our fancy new constraint

        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def solve(grid_chars):
    """
    Expects a grid in characters
    """

    # uses search to solve
     
    return search(grid_values(grid_chars))
    pass

def search(values):
    "Using depth-first search and propagation, create a search tree and solve the sudoku."

    # First, reduce the puzzle using the previous function
    values=reduce_puzzle(values)

    if values==False:
        return False
    
    # identifying the minimum length values > 1 in the grid
    moreThanOne = [box for box in values if len(values[box])>1]
    if len(moreThanOne)<1:
        return values
    min_val=float("infinity")
    min_box=''

    for i in moreThanOne:
        if len(values[i])<min_val:
            min_val=len(values[i])
            min_box=i

    for val in values[min_box]:
        new_values=values.copy() # make a deep copy as recursion may modify the old value
        new_values[min_box]=val    
        temp = search(new_values)
        if temp:
            return temp

if __name__ == '__main__':

   
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'     
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
