#solve_stochastic_tetris.py
from solve_tetris import build_reachable_mdp
from BabyTetris_02 import StochasticTetris

def minimax_value_iteration(env: StochasticTetris, mdp_data, gamma=None, theta=1e-8, max_iters=10000):
    """
    Minimax Value iteration on reachable set.
    Return V (dict state->value) and policy (dict state->best_action_index or None).
    """
    states = mdp_data['states']
    V = {s: 0.0 for s in states}
    policy = {s: None for s in states}

    for i in range(max_iters):
        delta = 0.0
        for s in states:
            v = V[s]
            max_min_value = float('-inf')
            best_action = None
            for a in env.get_actions(s):
                min_value = env.Q_minimax(s, a, gamma if gamma is not None else env.discount, V)
                if min_value > max_min_value:
                    max_min_value = min_value
                    best_action = a
            V[s] = max_min_value
            policy[s] = best_action
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    return V, policy

def extract_policies(env, V, states, gamma):
    """
    Extrait les politiques optimales.
    """
    policy_player = {}
    policy_adversary = {}
    
    for s in states:
        grid, _ = s
        
        best_action = None
        best_value = -float('inf')
        best_adv_response = None
        
        # Tester toutes les actions
        for action in range(12):
            # Vérifier validité
            valid_for_line = action in env.get_actions((grid, 0))
            valid_for_angle = action in env.get_actions((grid, 1))
            
            if not (valid_for_line or valid_for_angle):
                continue
            
            # Trouver la pire réponse adversaire
            worst_value = float('inf')
            worst_adv_choice = None
            
            for adv_choice in [0, 1]:
                # Vérifier si action valide avec cette pièce
                if adv_choice == 0 and not valid_for_line:
                    continue
                if adv_choice == 1 and not valid_for_angle:
                    continue
                
                # Simuler
                next_state, reward = env.getAdversarial_transition(s, action, adv_choice)
                V_next = V.get(next_state, 0.0)
                value = reward + gamma * V_next
                
                if value < worst_value:
                    worst_value = value
                    worst_adv_choice = adv_choice
            
            # Vérifier si meilleur pour le joueur
            if worst_value > best_value:
                best_value = worst_value
                best_action = action
                best_adv_response = worst_adv_choice
        
        policy_player[s] = best_action
        if best_action is not None:
            policy_adversary[(s, best_action)] = best_adv_response
    
    return policy_player, policy_adversary