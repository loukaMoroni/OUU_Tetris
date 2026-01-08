# analyzer.py
"""
Analyseur des politiques et résultats du jeu stochastique.
"""

class PolicyAnalyzer:
    """Analyse les politiques optimales du jeu."""
    
    def __init__(self, discount=0.95):
        self.discount = discount
    
    def is_terminal_grid(self, grid):
        """Vérifie si une grille est terminale."""
        return (grid & 0xF000) != 0
    
    def print_policy_summary(self, V, adversary_policy, player_policy):
        """Affiche un résumé des politiques."""
        start_grid = 0x0000
        
        print(f"\nPOLITIQUES (état initial, grille vide):")
        adv_choice = adversary_policy.get(start_grid)
        adv_name = "barre" if adv_choice == 0 else "angle" if adv_choice == 1 else "N/A"
        print(f"  Adversaire choisit: pièce {adv_choice} ({adv_name})")
        
        if start_grid in player_policy:
            for piece in [0, 1]:
                action = player_policy[start_grid].get(piece)
                if action is not None:
                    piece_name = "barre" if piece == 0 else "angle"
                    print(f"  Joueur (si pièce={piece_name}): action {action}")
    
    def analyze_adversary_choices(self, adversary_policy):
        """Analyse la distribution des choix de l'adversaire."""
        adv_choices = [p for p in adversary_policy.values() if p is not None]
        
        if not adv_choices:
            print("  Aucun choix d'adversaire à analyser.")
            return
        
        barre_count = sum(1 for p in adv_choices if p == 0)
        angle_count = len(adv_choices) - barre_count
        
        print(f"\nDistribution des choix adversaire:")
        print(f"  Barre: {barre_count} états ({100*barre_count/len(adv_choices):.1f}%)")
        print(f"  Angle: {angle_count} états ({100*angle_count/len(adv_choices):.1f}%)")
    
    def analyze_values(self, V):
        """Analyse les valeurs optimales."""
        non_terminal = [g for g in V if not self.is_terminal_grid(g)]
        
        if not non_terminal:
            print("  Aucune valeur à analyser.")
            return
        
        min_val = min(V[g] for g in non_terminal)
        max_val = max(V[g] for g in non_terminal)
        avg_val = sum(V[g] for g in non_terminal) / len(non_terminal)
        
        print(f"\nAnalyse des valeurs optimales:")
        print(f"  Minimum: {min_val:.4f}")
        print(f"  Maximum: {max_val:.4f}")
        print(f"  Moyenne: {avg_val:.4f}")
        
        # États terminaux
        terminal = [g for g in V if self.is_terminal_grid(g)]
        print(f"  États terminaux: {len(terminal)}")
    
    def compare_values(self, mdp_value, stochastic_value):
        """Compare les valeurs MDP et stochastique."""
        print(f"\nCOMPARAISON DES VALEURS:")
        print(f"  Valeur MDP (Question 1): {mdp_value:.6f}")
        print(f"  Valeur jeu stochastique (Question 2): {stochastic_value:.6f}")
        
        difference = mdp_value - stochastic_value
        print(f"  Différence: {difference:.6f}")
        
        if mdp_value > 0:
            reduction = 100 * difference / mdp_value
            print(f"  Réduction due à l'adversaire: {reduction:.2f}%")
        
        return reduction
    
    def export_stochastic_policies_simple(self, V, adv_policy, player_policy, 
                                         filename="policy_q2.py"):
        """Exporte les politiques du jeu stochastique dans un format simple."""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# POLITIQUES OPTIMALES JEU STOCHASTIQUE (QUESTION 2)\n")
            f.write("# " + "="*70 + "\n\n")
            
            # 1. Valeurs (optionnel)
            f.write("# Valeurs optimales V*(grille)\n")
            f.write("V = {\n")
            for grid, value in sorted(V.items()):
                f.write(f"    {hex(grid)}: {value:.6f},\n")
            f.write("}\n\n")
            
            # 2. Politique adversaire
            f.write("# Politique adversaire: grille → pièce (0=barre, 1=angle)\n")
            f.write("adv_policy = {\n")
            for grid, piece in sorted(adv_policy.items()):
                if piece is not None:  # Ignorer terminaux
                    piece_name = "barre" if piece == 0 else "angle"
                    f.write(f"    {hex(grid)}: {piece},  # {piece_name}\n")
            f.write("}\n\n")
            
            # 3. Politique joueur
            f.write("# Politique joueur: grille → {barre: action, angle: action}\n")
            f.write("player_policy = {\n")
            for grid, actions_dict in sorted(player_policy.items()):
                if actions_dict:  # Ignorer les dictionnaires vides
                    f.write(f"    {hex(grid)}: {{\n")
                    
                    for piece in [0, 1]:
                        action = actions_dict.get(piece)
                        if action is not None:
                            piece_name = "barre" if piece == 0 else "angle"
                            f.write(f"        {piece}: {action},  # {piece_name}\n")
                    
                    f.write("    },\n")
            f.write("}\n\n")
            
            # 4. Résumé
            f.write("# " + "="*70 + "\n")
            f.write(f"# Résumé:\n")
            f.write(f"# - États totaux: {len(V)}\n")
            f.write(f"# - Politique adversaire: {len([p for p in adv_policy.values() if p is not None])} états\n")
            f.write(f"# - Politique joueur: {len(player_policy)} grilles avec actions\n")
            f.write("# " + "="*70 + "\n")
        
        print(f"✓ Politiques jeu stochastique exportées: {filename}")
        return filename

    def print_conclusion(self, reduction, mdp_reward, stochastic_reward):
        """Affiche la conclusion de l'analyse."""
        sim_difference = mdp_reward - stochastic_reward
        sim_reduction = 100 * sim_difference / mdp_reward if mdp_reward > 0 else 0
        
        print("\n" + "="*70)
        print("CONCLUSION:")
        print(f"L'adversaire optimal réduit le score:")
        print(f"  - En théorie (valeur optimale): {reduction:.2f}%")
        print(f"  - En pratique (simulation): {sim_reduction:.2f}%")
        print("par rapport au cas où la pièce est aléatoire.")
        print("="*70)