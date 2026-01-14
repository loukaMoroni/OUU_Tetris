# simulator.py

import random
from solve_tetris import simulate_policy

class GameSimulator:
    """Simule des parties du jeu stochastique."""
    
    def __init__(self, discount=0.95):
        self.discount = discount
    
    def simulate_game(self, adversary_policy, player_policy, transitions, 
                     max_steps=100, seed=None):
        """Simule une partie complète avec affichage détaillé.
        """
        if seed is not None:
            random.seed(seed)
        
        # Pré-calcul des actions pour optimisation
        actions_map = {}
        for (grid, piece, action) in transitions:
            if (grid, piece) not in actions_map:
                actions_map[(grid, piece)] = []
            actions_map[(grid, piece)].append((action, transitions[(grid, piece, action)]))
        
        # Initialisation
        grid = 0x0000
        total_discounted = 0.0
        steps = 0
        
        print(f"\nÉtat initial:")
        self.print_grid(grid)
        
        # Boucle de simulation
        while steps < max_steps and not self.is_terminal_grid(grid):
            print(f"\n{'─'*40}")
            print(f"Étape {steps}:")
            
            # 1. Adversaire choisit la pièce
            piece = adversary_policy.get(grid)
            if piece is None:
                print("  État terminal, fin du jeu.")
                break
            
            # 2. Joueur choisit l'action
            action = self._get_player_action(grid, piece, player_policy, actions_map)
            if action is None:
                break
            
            # 3. Exécution de l'action
            result = self._execute_action(grid, piece, action, transitions, actions_map)
            if result is None:
                break
            
            next_grid, reward = result
            
            # 4. Calcul des récompenses
            step_discounted = (self.discount ** steps) * reward
            total_discounted += step_discounted
            
            # 5. Affichage des résultats
            self._display_step_results(piece, action, reward, step_discounted, next_grid)
            
            # 6. Passage à l'état suivant
            grid = next_grid
            steps += 1
        
        # Affichage des résultats finaux
        self._display_final_results(steps, total_discounted, grid)
        
        return total_discounted, steps, grid
    
    def _get_player_action(self, grid, piece, player_policy, actions_map):
        """Récupère l'action du joueur."""
        action = player_policy.get(grid, {}).get(piece)
        if action is None and (grid, piece) in actions_map:
            action_info = random.choice(actions_map[(grid, piece)])
            action = action_info[0]
        
        if action is None:
            print("  Aucune action possible, fin du jeu.")
        
        return action
    
    def _execute_action(self, grid, piece, action, transitions, actions_map):
        """Exécute une action et retourne le résultat."""
        if (grid, piece, action) in transitions:
            return transitions[(grid, piece, action)]
        else:
            # Recherche dans actions_map
            for a_info in actions_map.get((grid, piece), []):
                if a_info[0] == action:
                    return a_info[1]
            
            print("  Transition non trouvée, fin du jeu.")
            return None
    
    def _display_step_results(self, piece, action, reward, step_discounted, next_grid):
        """Affiche les résultats d'une étape."""
        print(f"  Adversaire: pièce {piece} ({'barre' if piece == 0 else 'angle'})")
        print(f"  Joueur: action {action}")
        print(f"  Récompense: {reward} (actualisée: {step_discounted:.4f})")
        print(f"\n  Nouvelle grille:")
        self.print_grid(next_grid)
        
        if self.is_terminal_grid(next_grid):
            print("Terminal!")
    
    def _display_final_results(self, steps, total_discounted, final_grid):
        """Affiche les résultats finaux."""
        print(f"\n{'='*60}")
        print(f"FIN après {steps} étapes")
        print(f"Récompense totale actualisée: {total_discounted:.4f}")
        print(f"Grille finale:")
        self.print_grid(final_grid)
    
    def is_terminal_grid(self, grid):
        """Vérifie si une grille est terminale."""
        return (grid & 0xF000) != 0
    
    def print_grid(self, grid, title="Grid"):
        """Affiche une grille 4x4."""
        bits = f"{grid:016b}"
        print(f"{title} ({hex(grid)}):")
        for row in range(4):
            row_bits = bits[row*4:(row+1)*4]
            row_chars = []
            for col in range(4):
                bit = row_bits[3-col]
                row_chars.append('1' if bit == '1' else '.')
            print(f"  {' '.join(row_chars)}")
    
    def simulate_mdp_for_comparison(self, env, policy_mdp, max_steps=30, seed=42):
        """Simule une partie MDP pour comparaison."""
        returns = simulate_policy(env, policy_mdp, episodes=1, 
                                 max_steps=max_steps, seed=seed, render=False)
        return returns[0]