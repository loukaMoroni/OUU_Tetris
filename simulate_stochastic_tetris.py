# simulate_stochastic_tetris.py
import random
from BabyTetris_02 import StochasticTetris
from solve_stochastic_tetris import minimax_value_iteration, extract_policies
from solve_tetris import build_reachable_mdp

def simulate_adversarial_policy(env: StochasticTetris, 
                                player_policy, 
                                adversary_policy, 
                                episodes=10, 
                                max_steps=200, 
                                seed=None, 
                                render=False):
    """
    Simule la politique adversarial pour plusieurs épisodes.
    Retourne la liste des retours (un par épisode).
    """
    if seed is not None:
        random.seed(seed)
    
    gamma = env.get_discount_factor()
    returns = []
    
    for ep in range(episodes):
        state = env.get_initial_state()
        env.state = state
        
        G = 0.0  # Gain total discounté
        t = 0
        trajectory = []
        
        while t < max_steps and not env.is_terminal(state):
            # 1. Le joueur choisit son action selon sa politique
            player_action = player_policy.get(state)
            if player_action is None:
                break
            
            # Vérifier que l'action est valide (sécurité)
            if player_action not in env.get_actions(state):
                player_action = random.choice(list(env.get_actions(state)))
            
            # 2. L'adversaire répond selon sa politique
            adversary_choice = adversary_policy.get((state, player_action))
            if adversary_choice is None:
                # Si pas de stratégie spécifique, l'adversaire choisit au hasard
                adversary_choice = random.choice([0, 1])
            
            # 3. Simuler la transition
            next_state, reward = env.getAdversarial_transition(
                state, player_action, adversary_choice
            )
            
            # 4. Mettre à jour le gain
            G += (gamma ** t) * reward
            
            # 5. Enregistrer pour affichage
            if render:
                trajectory.append({
                    'state': state,
                    'player_action': player_action,
                    'adversary_choice': adversary_choice,
                    'reward': reward,
                    'next_state': next_state
                })
            
            # 6. Passer à l'état suivant
            state = next_state
            env.state = state
            t += 1
        
        returns.append(G)
        
        if render:
            print(f"\n=== Épisode {ep+1} (étapes: {t}) ===")
            print(f"Gain total discounté: {G:.4f}")
            
            for step, data in enumerate(trajectory):
                print(f"\nÉtape {step}:")
                print(f"  État: {data['state']}")
                print(f"  Action joueur: {data['player_action']}")
                print(f"  Choix adversaire: {'LIGNE' if data['adversary_choice']==0 else 'ANGLE'}")
                print(f"  Récompense: {data['reward']}")
                print(f"  État suivant: {data['next_state']}")
            
            print("-" * 50)
    
    return returns

def display_grid_4x4(grid):
    """Affiche la grille sur 4 lignes."""
    bits = f"{grid:016b}"
    for i in range(0, 16, 4):
        line = bits[i:i+4].replace('0', '.').replace('1', '#')
        print(line)

def simulate_with_visualization():
    """
    Simulation complète avec visualisation.
    """
    print("=" * 60)
    print("SIMULATION TETRIS AVEC ADVERSARIE (Q2)")
    print("=" * 60)
    
    # 1. Créer l'environnement
    env = StochasticTetris(discount=0.95)
    print("✓ Environnement créé")
    
    # 2. Explorer les états
    print("\nExploration des états atteignables (BFS)...")
    mdp_data = build_reachable_mdp(env, max_states=200000)
    states = mdp_data['states']
    print(f"✓ États atteignables: {len(states)}")
    
    # 3. Calculer la valeur et politiques optimales
    print("\nCalcul des valeurs optimales (Minimax Value Iteration)...")
    V, policy_player = minimax_value_iteration(
        env, mdp_data, gamma=0.95, theta=1e-7
    )
    
    print("\nExtraction des politiques optimales...")
    policy_player_full, policy_adversary = extract_policies(
        env, V, states, 0.95
    )
    
    # 4. Afficher résultats pour état initial
    init_state = env.get_initial_state()
    print(f"\n=== RÉSULTATS POUR ÉTAT INITIAL ===")
    print(f"État: {init_state}")
    print(f"Valeur optimale: {V[init_state]:.4f}")
    print(f"Meilleure action joueur: {policy_player[init_state]}")
    
    if policy_player[init_state] is not None:
        adv_key = (init_state, policy_player[init_state])
        if adv_key in policy_adversary:
            adv_choice = policy_adversary[adv_key]
            piece_name = "LIGNE" if adv_choice == 0 else "ANGLE"
            print(f"Réponse optimale adversaire: {piece_name}")
    
    # 5. Simuler quelques parties
    print("\n" + "=" * 60)
    print("SIMULATION DE 3 PARTIES")
    print("=" * 60)
    
    returns = simulate_adversarial_policy(
        env, policy_player, policy_adversary,
        episodes=3, max_steps=50, seed=42, render=True
    )
        
    return V, policy_player, policy_adversary, returns


if __name__ == "__main__":
    print(" Simulation complète")   
    V, policy_player, policy_adversary, returns = simulate_with_visualization()   
    print("\nSimulation terminée.")