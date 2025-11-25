import random

class BabyTetris():
    
    discount_factor: float 
    initial_state: int
    terminal_states_max: int
    step: int 
    total_rewards: float
    line: int[6]
    bend: int[12]
    state: tuple # grid state
    actions: tuple

    def __init__(self, discount):
        self.discount_factor = discount
        self.terminal_states_max= 0xFFFF
        self.step = 0
        self.total_rewards = 0.0
        self.initial_state = 0x0
        self.state = (self.initial_state, 0)

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

        self.actions(self.line, self.bend)



    # Return all states of this MDP
    def get_states(self):
        """ 
        The states are represented as an integer from 0 to 2**16
        each bit of th integer representing a cell in the baby tetris grid 
        the bit being set to 1 meaning that a block is present in the cell and 0 otherwise.
        And another integer to represent the nex piece about to fall

        this allow us, if we read the integer in hexadecimal to see each digit as a line. 
        One of the digits being F meaning the line is full 
        """
        return range(0,0x10000)

    # Return all actions with non-zero probability from this state 
    def get_actions(self, state):
        if (state[1]==0):       # if the piece is the line
            return range(0,6)   # we return the six posible positions
        else :                  # if it's the bend
            return range(0,12)  #we return the twelve possible positions
        
        #return self.actions[state[1]]

    def get_transitions(self, state, action):
        
        
        grid = compute_grid(self, state, action)

        if(is_terminal(state)):
            return [(state,1)]
        
        #testing for completed lines and lowering if needed
        if (grid &  0xF000) == 0xF000:
            grid &= 0x0FFF #remove line
        if (state[0] &  0x0F00) == 0x0F00:
            top = grid & 0xF000 #stock above line
            grid &= 0xF0FF #remove line
            grid |= (top>>4) #lower above line
        if (state[0] &  0x00F0) == 0x00F0:
            top = grid & 0xFF00
            grid &= 0xFF0F 
            grid |= (top>>4) 
        if (state[0] & 0x000F) == 0x000F:
            top = grid & 0xFFF0
            grid &= 0xFFF0 
            grid |= (top>>4) 

        line = (state[0],0)
        bend = (state[0],1)

        #we assume uniform distribution over the pieces
        return [(line,0.5),(bend,0.5)]


    def lower_piece(self, state, piece, floor):
        piece_on_grid = piece
        row=0
        # while the piece doesn't intersect an existing bloc or go out of the grid
        while(state & piece_on_grid !=0 and row<floor) : 
            # we lower the piece by one row 
            piece_on_grid = piece_on_grid>>4
            row+=1
        if (row==0):
            return piece<<4
        return piece>>4*(row-1)
    
    
    def compute_grid(self, state, action):
        #                     piece  position
        piece = self.actions[state[1]][action]

        if (state[1]==1):
            piece = lower_piece(state[0],piece, 3)
        elif (action <2):
            piece = lower_piece(state[0],piece, 2)
        else:
            piece = lower_piece(state[0],piece, 4)

        return state[0]+piece
    

    def get_reward(self, state, action):
        reward = (0,1,3,6)
        nb_line = 0
        
        state[0] = compute_grid(state, action)

        #testing for completed lines
        if (state[0] &  0xF000) == 0xF000:
            nb_line+=1
        if (state[0] &  0x0F00) == 0x0F00:
            nb_line+=1
        if (state[0] &  0x00F0) == 0x00F0:
            nb_line+=1
        if (state[0] & 0x000F) == 0x000F:
            nb_line+=1
        
        reward = reward[nb_line]

        #step +=1
        #self.total_rewards += reward * (self.discount_factor ** step)
        return reward
    
    

    # Return true if and only if state is a terminal state of this MDP 
    def is_terminal(self, state):
    
        return (state[0] > self.terminal_states_max)

    # Return the discount factor for this MDP
    def get_discount_factor(self):
        return self.discount_factor

    # Return the initial state of this MDP
    def get_initial_state(self):
        return self.initial_state
