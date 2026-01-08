# main_stochastic.py
"""
Programme principal à exécuter pour la Question 2.
"""

from stochastic_solver import StochasticSolver
from simulator import GameSimulator
from analyzer import PolicyAnalyzer
from BabyTetris import BabyTetris
from solve_tetris import build_reachable_mdp, value_iteration

def print_header():
    """Affiche l'en-tête du programme."""
    print("="*70)
    print("QUESTION 2: JEU STOCHASTIQUE TETRIS")
    print("="*70)

def main():
    print_header()
    
    # 1. CRÉER LE MÊME ENVIRONNEMENT
    env = BabyTetris(discount=0.95)
    solver = StochasticSolver(discount=0.95)
    simulator = GameSimulator(discount=0.95)
    analyzer = PolicyAnalyzer(discount=0.95)
    
    # 2. RÉUTILISER LA BFS EXISTANTE
    print("\n[Étape 1] Utilisation de build_reachable_mdp du code existant...")
    mdp_data = build_reachable_mdp(env, max_states=200000)
    print(f"États MDP atteignables: {len(mdp_data['states'])}")
    
    # 3. RÉUTILISER LES TRANSITIONS EXISTANTES
    print("\n[Étape 2] Construction des transitions du jeu stochastique...")
    
    # 4. RÉSOUDRE LE JEU STOCHASTIQUE
    print("\n[Étape 3] Résolution du jeu stochastique (min-max)...")
    V_stochastic, adv_policy, player_policy, transitions = solver.solve_stochastic_game(mdp_data)
    
    # 5. COMPARAISON AVEC QUESTION 1
    print("\n[Étape 4] Comparaison avec la Question 1 (MDP)...")
    
    # Pour la Question 1, utilise value_iteration existante
    V_mdp, policy_mdp = value_iteration(env, mdp_data, gamma=env.get_discount_factor())
    
    start_state = env.get_initial_state()
    start_grid = start_state[0]
    
    # Valeur MDP (moyenne sur les pièces)
    mdp_value = 0.0
    for piece in [0, 1]:
        state = (start_grid, piece)
        if state in V_mdp:
            mdp_value += V_mdp[state]
    mdp_value /= 2
    
    # Valeur jeu stochastique
    stochastic_value = V_stochastic.get(start_grid, 0.0)
    
    # Analyse de la comparaison
    reduction = analyzer.compare_values(mdp_value, stochastic_value)
    
    # 6. AFFICHER LES POLITIQUES
    analyzer.print_policy_summary(V_stochastic, adv_policy, player_policy)
    
    # 7. SIMULATION
    print("\n" + "="*70)
    print("SIMULATION DÉTAILLÉE")
    print("="*70)
    
    total_reward, steps, final_grid = simulator.simulate_game(
        adv_policy, player_policy, transitions,
        max_steps=30,
        seed=42
    )
    
    # 8. SIMULATION MDP POUR COMPARAISON
    print("\n" + "="*70)
    print("SIMULATION MDP (Question 1) POUR COMPARAISON")
    print("="*70)
    
    # Réutilise simulate_policy du code existant
    mdp_reward = simulator.simulate_mdp_for_comparison(env, policy_mdp, max_steps=30, seed=42)
    print(f"Retour MDP: {mdp_reward:.4f}")
    print(f"Retour jeu stochastique: {total_reward:.4f}")
    
    # 9. ANALYSE DÉTAILLÉE
    print("\n" + "="*70)
    print("ANALYSE DÉTAILLÉE")
    print("="*70)
    
    # Analyse des choix de l'adversaire
    analyzer.analyze_adversary_choices(adv_policy)
    
    # Analyse des valeurs
    analyzer.analyze_values(V_stochastic)
    
    # 10. CONCLUSION
    analyzer.print_conclusion(reduction, mdp_reward, total_reward)
    # 11. EXPORT SIMPLE DES POLITIQUES
    print("\n" + "="*70)
    print("EXPORT SIMPLE DES POLITIQUES")
    print("="*70)
    
    # Export Question 2
    analyzer.export_stochastic_policies_simple(
        V_stochastic, adv_policy, player_policy,
        filename="question2_policy.py"
    )
        
    print("✓ Fichiers généré:")
    print("  - question2_policy.py  (politique jeu stochastique)")

if __name__ == "__main__":
    main()