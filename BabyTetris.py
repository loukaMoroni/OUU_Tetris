#BabyTetris.py
class BabyTetris():
    discount_factor: float 
    initial_state: int
    terminal_states_max: int
    step: int 
    total_rewards: float
    state: tuple
    actions: tuple

    def __init__(self, discount):
        import random
        self.discount_factor = discount
        self.terminal_states_max= 0xFFFF
        self.step = 0
        self.total_rewards = 0.0
        self.initial_state = (0x0000, random.randint(0,1))  # empty grid and the  piece to fall
        self.state = self.initial_state
        # the line tetris piece in the 6 possibles positions represented in hexadecimal on a binary 4*4 grid  
        #        ###.   .###   #...   .#..   ..#.    ...#
        #        ....   ....   #...   .#..   ..#.    ...#
        #        ....   ....   #...   .#..   ..#.    ...#
        #        ....   ....   ....   ....   ....    ....
        self.line = (0xE000,0x7000,0X8880,0x4440,0x2220, 0X1110)

        # the bend tetris piece in the 12 possible posissions represented in hexadecimal on a binary grid 4*4
        # it a conbination of four rotation and three displacment
        
        #        .#..   #...   ##..   ##..   ..#.   .#..   .##.   .##.   ...#   ..#.   ..##   ..##
        #        ##..   ##..   #...   .#..   .##.   .##.   .#..   ..#.   ..##   ..##   ..#.   ...#
        #        ....   ....   ....   ....   ....   ....   ....   ....   ....   ....   ....   ....
        #        ....   ....   ....   ....   ....   ....   ....   ....   ....   ....   ....   ....
        self.bend = (0x4C00,0x8C00,0xC800,0xC400,0x2600,0x4600,0x6400,0x6200,0x1300,0x2300,0x3200,0x3100)

        self.actions = (self.line, self.bend)




    # Return all states of this MDP
    def get_states(self):
        """Generate all possible states of the MDP.
        """
        return ((g, p) for g in range(0x10000) for p in (0,1))


    # Return all actions with non-zero probability from this state 
    def get_actions(self, state):
        if (state[1]==0):       # if the piece is the line
            return range(0,6)   # we return the six posible positions
        else :                  # if it's the bend
            return range(0,12)  #we return the twelve possible positions
        
        #return self.actions[state[1]]
    def clear_full_lines(self, grid):
        """Identify and clear any full 4-bit line."""
        for shift in (12, 8, 4, 0):
            mask_line = 0xF << shift
            if (grid & mask_line) == mask_line:

                # bits above this line
                above_mask = (~((1 << (shift + 4)) - 1)) & 0xFFFF
                above = grid & above_mask

                # remove the line
                grid &= (~mask_line) & 0xFFFF

                # drop above content by one line (4 bits)
                above >>= 4

                # reconstruct grid
                grid = (grid & (~above_mask)) | above | (grid & ((1 << shift) - 1))

        return grid & 0xFFFF

    def get_transitions(self, state, action):
        """Return list of (next_state, prob) pairs for this state and action."""
        if self.is_terminal(state):
            return [(state,1.0)]
        new_grid,failed=self.compute_grid(state,action)
        if failed:
            return [(state,1.0)]
        grid_after=self.clear_full_lines(new_grid)
        #next piece uniform
        return[((grid_after,0),0.5),((grid_after,1),0.5)]


    def lower_piece(self, grid, piece, max_drop):
        """Lower the piece until it collides or reaches max drop."""
        piece_on_grid = piece
        row=0
        # while the piece doesn't intersect an existing bloc or go out of the grid
        while (grid & piece_on_grid) == 0 and row < max_drop:
            piece_on_grid >>= 4
            row += 1
        # collision immediately → impossible to place
        if row == 0 and (grid & piece_on_grid) != 0:
            return piece_on_grid, True

        # If collision, back up one row
        if (grid & piece_on_grid) != 0:
            piece_on_grid <<= 4

        return piece_on_grid, False
   
    
    def compute_grid(self, state, action):
        """Compute the new grid after placing the piece for the given action."""
        # Place piece,apply drop logic
        grid,piece_type=state
        piece = self.actions[piece_type][action]
        if piece_type == 1:
            max_drop=3
        elif action < 2:
            max_drop=2
        else:
            max_drop=4
        piece_on_grid,failed=self.lower_piece(grid,piece,max_drop)
        if failed:
            return grid,True
        new_grid=grid|piece_on_grid
        return new_grid,False

    


    def get_reward(self, state, action):
        """Return the reward for taking this action in this state."""
        rewards = (0, 1, 3, 6)
        grid,failed = self.compute_grid(state, action)
        if failed:
            return 0.0
        #count full lines before clear
        nb_line = 0
        if (grid & 0xF000) == 0xF000: nb_line += 1
        if (grid & 0x0F00) == 0x0F00: nb_line += 1
        if (grid & 0x00F0) == 0x00F0: nb_line += 1
        if (grid & 0x000F) == 0x000F: nb_line += 1
        reward = rewards[min(nb_line, 3)]
        return float(reward)

    
    

    # Return true if and only if state is a terminal state of this MDP 
    def is_terminal(self, state):
        """A state is terminal if the grid is full at the top line."""
        # grid, _ = state
        # return (grid & 0xF000) != 0
        return False
    
    # Return the discount factor for this MDP
    def get_discount_factor(self):
        return self.discount_factor

    # Return the initial state of this MDP
    def get_initial_state(self):
        return self.initial_state