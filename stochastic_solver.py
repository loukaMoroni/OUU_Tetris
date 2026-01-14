# stochastic_solver.py
"""
Solveur pour le jeu stochastique Tetris (Question 2).
Réutilise complètement le code de la Question 1.
"""

from solve_tetris import build_reachable_mdp, value_iteration

class StochasticSolver:
    """Classe qui résout le jeu stochastique Tetris."""
    
    def __init__(self, discount=0.95):
        from BabyTetris import BabyTetris
        self.env = BabyTetris(discount)
        self.discount = discount
    
    def extract_grids_from_mdp(self, mdp_data):
        """Extrait toutes les grilles uniques des états MDP."""
        grids = set()
        for state in mdp_data['states']:
            grid, _ = state
            grids.add(grid)
        return list(grids)
    
    def build_transitions_from_mdp(self, mdp_data):
        """
        Construit les transitions déterministes du jeu stochastique.
        Transforme les transitions probabilistes du MDP en transitions déterministes.
        """
        transitions = {}
        
        for state in mdp_data['transitions']:
            grid, piece = state
            
            for (action, next_state, prob) in mdp_data['transitions'][state]:
                # next_state est (next_grid, next_piece)
                next_grid, _ = next_state
                
                # La récompense est déjà calculée dans le MDP
                reward = mdp_data['rewards'].get((state, action), 0.0)
                
                # Stocke la transition déterministe
                transitions[(grid, piece, action)] = (next_grid, reward)
        
        return transitions
    
    def solve_stochastic_game(self, mdp_data, theta=1e-6, max_iters=1000):
        """
        Résout le jeu stochastique par value iteration avec min-max.
        
        Équation de Bellman:
        V(grid) = min_{piece ∈ {0,1}} [ max_{action} ( reward + γ * V(next_grid) ) ]
        
        Returns:
            tuple: (V, adversary_policy, player_policy, transitions)
        """
        # Extraire les données du MDP
        grids = self.extract_grids_from_mdp(mdp_data)
        transitions = self.build_transitions_from_mdp(mdp_data)
        
        # Optimisation: pré-calcul des actions
        actions_map = {}
        for (grid, piece, action) in transitions:
            if (grid, piece) not in actions_map:
                actions_map[(grid, piece)] = []
            actions_map[(grid, piece)].append((action, transitions[(grid, piece, action)]))
        
        # Initialisation des valeurs
        V = {grid: 0.0 for grid in grids}
        
        # Value iteration avec min-max
        for iteration in range(max_iters):
            delta = 0.0
            
            for grid in grids:
                if self.is_terminal_grid(grid):
                    continue
                
                # MIN (adversaire choisit la pièce)
                min_value = float('inf')
                
                for piece in [0, 1]:
                    # MAX (joueur choisit l'action)
                    max_value = -float('inf')
                    
                    if (grid, piece) in actions_map:
                        for action, (next_grid, reward) in actions_map[(grid, piece)]:
                            value = reward + self.discount * V.get(next_grid, 0.0)
                            if value > max_value:
                                max_value = value
                    else:
                        max_value = 0.0
                    
                    if max_value < min_value:
                        min_value = max_value
                
                delta = max(delta, abs(V[grid] - min_value))
                V[grid] = min_value
            
            if delta < theta:
                print(f"Convergence à l'itération {iteration}, delta={delta}")
                break
            
            if iteration % 50 == 0:
                print(f"  Itération {iteration}, delta={delta}")
        
        # Extraction des politiques optimales
        adversary_policy, player_policy = self._extract_policies(V, actions_map)
        
        return V, adversary_policy, player_policy, transitions
    
    def _extract_policies(self, V, actions_map):
        """Extrait les politiques optimales à partir des valeurs V."""
        adversary_policy = {}
        player_policy = {}
        #Parcours de toutes les grilles possibles dans la fonction de valeur V
        for grid in V:
            #Si le jeu est fini, pas de politique à extraire
            if self.is_terminal_grid(grid):
                adversary_policy[grid] = None #l'adversaire ne choisit plus de pièces
                player_policy[grid] = {0: None, 1: None} #Le joueur ne peut plus jouer
                continue
            
            best_piece = None #Meilleure pièce pour l'adversaire
            best_val = float('inf')#Meilleure valeur associée à la meilleure pièce
            
            for piece in [0, 1]:
                player_best = -float('inf') #Meilleure valeur pour le joueur
                best_action = None
                
                if (grid, piece) in actions_map:
                    #le joueur évalue toutes les actions possibles pour la pièce donnée
                    for action, (next_grid, reward) in actions_map[(grid, piece)]:
                        #Calcul de la valeur Q: récompense immédiate + valeur future
                        #Formule: Q=reward + γ * V(next_grid)
                        value = reward + self.discount * V.get(next_grid, 0.0)
                        if value > player_best:
                            player_best = value
                            best_action = action
                else:
                    player_best = 0.0
                
                if grid not in player_policy:
                    player_policy[grid] = {}
                player_policy[grid][piece] = best_action
                
                if player_best < best_val:
                    best_val = player_best
                    best_piece = piece
            
            adversary_policy[grid] = best_piece
        
        return adversary_policy, player_policy
    
    def is_terminal_grid(self, grid):
        """Vérifie si une grille est terminale (ligne du haut remplie)."""
        return (grid & 0xF000) != 0
    
    def compare_with_mdp(self, mdp_data, V_stochastic):
        """
        Compare la valeur du jeu stochastique avec celle du MDP (Question 1).
        
        Returns:
            tuple: (mdp_value, stochastic_value)
        """
        # Calculer la valeur MDP
        V_mdp, _ = value_iteration(self.env, mdp_data, gamma=self.discount)
        
        start_grid = self.env.get_initial_state()[0]
        
        # Valeur MDP = moyenne sur les 2 pièces possibles
        mdp_value = 0.0
        for piece in [0, 1]:
            state = (start_grid, piece)
            if state in V_mdp:
                mdp_value += V_mdp[state]
        mdp_value /= 2
        
        # Valeur jeu stochastique
        stochastic_value = V_stochastic.get(start_grid, 0.0)
        
        return mdp_value, stochastic_value