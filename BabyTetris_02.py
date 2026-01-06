#babyTetris_02.py
from BabyTetris import BabyTetris

class StochasticTetris(BabyTetris):
    def __init__(self, discount):
        super().__init__(discount)

    def get_next_piece(self):
        return self.random.choice(self.pieces)
    
    def getAdversarial_transition(self,state,player_action,adversary_choice):
        """
        Given a state, a player action, and an adversary choice (next piece),
        this function returns the next state after applying the action 
        and the adversary's choice.

        """
        #Extract the grid of the current state
        grid, _=state
        state_1=(grid,adversary_choice)
        if(player_action not in self.get_actions(state_1)):
            return (state,0.0) #return same state with 0 reward if action invalid

        #compute new grid after applying player action
        new_grid, failed = self.compute_grid(state_1, player_action)
        if failed:
            return (state,0.0) #return same state with 0 reward if action fails
        reward=self.get_reward(state_1,player_action)
        grid=self.clear_full_lines(new_grid)
        next_state=(grid, _)
        return (next_state,reward)
    
    def Q_minimax(self,state,player_action,gamma,V):
        """
        computes the min over adversary choices of:
        R(s,a_player,a_adversary) + gamma*V(s')

        """
        min_E=float('inf')
        for adversary_choice in [0,1]:
            next_state,reward=self.getAdversarial_transition(state,player_action,adversary_choice)
            E=reward+gamma*V.get(next_state,0.0)
            if E<min_E:
                min_E=E
        return min_E
    
