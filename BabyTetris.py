class BabyTetris():

    
    
    discount_factor: float 
    initial_state: int 
    terminal_states_boundaries: tuple 
    step: int 
    total_rewards: float

    def __init__(self, discount):
        self.discount_factor = discount
        self.initial_state = 0
        self.terminal_states_boundaries = ( -0xFFFF, 0xFFFF )
        self.step = 0
        total_rewards = 0.0


    # Return all states of this MDP
    def get_states(self):
        """ 
        The states are represented as the integer from -2**16 to 2**16
        each bit of th integer representing a cell in the baby tetris grid 
        the bit being set to 1 meaning that a block is present in the cell and 0 otherwise
        the sign bit represent wether the next tetris block is a line or a bend

        this allow us, if we read the integer in hexadecimal to see each digit as a line
        and the sign as the next block. One of the digits being F meaning the line is full 
        """
        return 

    # Return all actions with non-zero probability from this state 
    def get_actions(self, state):
        abstract

    def get_transitions(self, state, action):
        """
        transitions = []

        if state == self.TERMINAL:
            if action == self.TERMINATE:
                return [(self.TERMINAL, 1.0)]
            else:
                return []

        # Probability of not slipping left or right
        straight = 1 - (2 * self.noise)

        (x, y) = state
        if state in self.get_goal_states().keys():
            if action == self.TERMINATE:
                transitions += [(self.TERMINAL, 1.0)]

        elif action == self.UP:
            transitions += self.valid_add(state, (x, y + 1), straight)
            transitions += self.valid_add(state, (x - 1, y), self.noise)
            transitions += self.valid_add(state, (x + 1, y), self.noise)

        elif action == self.DOWN:
            transitions += self.valid_add(state, (x, y - 1), straight)
            transitions += self.valid_add(state, (x - 1, y), self.noise)
            transitions += self.valid_add(state, (x + 1, y), self.noise)

        elif action == self.RIGHT:
            transitions += self.valid_add(state, (x + 1, y), straight)
            transitions += self.valid_add(state, (x, y - 1), self.noise)
            transitions += self.valid_add(state, (x, y + 1), self.noise)

        elif action == self.LEFT:
            transitions += self.valid_add(state, (x - 1, y), straight)
            transitions += self.valid_add(state, (x, y - 1), self.noise)
            transitions += self.valid_add(state, (x, y + 1), self.noise)

        # Merge any duplicate outcomes
        merged = defaultdict(lambda: 0.0)
        for (state, probability) in transitions:
            merged[state] = merged[state] + probability

        transitions = []
        for outcome in merged.keys():
            transitions += [(outcome, merged[outcome])]

        return transitions
        """

    def valid_add(self, state, new_state, probability):
        """"
        # If the next state is blocked, stay in the same state
        if probability == 0.0:
            return []

        if new_state in self.blocked_states:
            return [(state, probability)]

        # Move to the next space if it is not off the grid
        (x, y) = new_state
        if x >= 0 and x < self.width and y >= 0 and y < self.height:
            return [((x, y), probability)]

        # If off the grid, state in the same state
        return [(state, probability)]
        """

    def get_reward(self, state, action, new_state):
        reward = (0,1,3,6)
        nb_line = 0
        #testing for complete lines
        if (state &  0xF000) == 0xF000:
            nb_line+=1
        if (state &  0x0F00) == 0x0F00:
            nb_line+=1
        if (state &  0x00F0) == 0x00F0:
            nb_line+=1
        if (state & 0x000F) == 0x000F:
            nb_line+=1
        
        reward = reward[nb_line]

        step +=1
        self.total_rewards += reward * (self.discount_factor ** step)
        return reward

    # Return true if and only if state is a terminal state of this MDP 
    def is_terminal(self, state):
        return (state < self.terminal_states_boundaries[0] or  state > self.terminal_states_boundaries[1])

    # Return the discount factor for this MDP
    def get_discount_factor(self):
        return self.discount_factor

    # Return the initial state of this MDP
    def get_initial_state(self):
        return self.initial_state
