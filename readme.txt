===============================================================================
                            PROJET BABY TETRIS
===============================================================================

Ce projet implémente un Tetris simplifié avec 2 types de pièces et résout
les problèmes d'optimisation contre un adversaire aléatoire (Question 1)
et un adversaire optimal (Question 2).

===============================================================================
📁 STRUCTURE DES FICHIERS
===============================================================================

FICHIERS PRINCIPAUX :
---------------------
1. BabyTetris.py        - Environnement de jeu (grilles, pièces, règles)
2. solve_tetris.py      - Question 1: MDP contre adversaire aléatoire
3. main_stochastic.py   - Question 2: Jeu stochastique contre adversaire optimal
4. analyzer.py          - Analyse et comparaison des résultats
5. simulator.py         - Simulation des parties
6. stochastic_solver.py - Résolution jeu stochastique

FICHIERS DE RÉFÉRENCE :
-----------------------
7. actions_legend.txt   - Légende des numéros d'actions (IMPORTANT!)
8. README.txt           - Ce fichier
9. test_tetris.py       - Tests basiques de l'environnement

FICHIERS GÉNÉRÉS (après exécution) :
-------------------------------------
10. question1_policy.py - Politique optimale Question 1
11. question2_policy.py - Politiques optimales Question 2

===============================================================================
🚀 COMMENT TESTER
===============================================================================

ÉTAPE 1 - VÉRIFICATION DE L'ENVIRONNEMENT :
-------------------------------------------
> python test_tetris.py
→ Devrait afficher: état initial, actions possibles, exemple de grille

ÉTAPE 2 - QUESTION 1 (MDP) :
----------------------------
> python solve_tetris.py

RÉSULTATS ATTENDUS :
- Exploration BFS: ~22,162 états atteignables
- Value iteration converge en ~161 itérations  
- Valeur optimale: ~18.18 (discount 0.95)
- Génère: question1_policy.py (politique optimale)

ÉTAPE 3 - QUESTION 2 (JEU STOCHASTIQUE) :
-----------------------------------------
> python main_stochastic.py

RÉSULTATS ATTENDUS :
- Réutilisation des 22,162 états de la Question 1
- Min-max converge en ~147 itérations
- Valeur optimale: ~14.13 (22% moins que Question 1)
- Génère: question2_policy.py (politiques adversaire + joueur)
- Simulation détaillée de 30 étapes

===============================================================================
📊 INTERPRÉTATION DES RÉSULTATS
===============================================================================

VALEURS OPTIMALES :
-------------------
• Question 1 (MDP):      ~18.18  (adversaire aléatoire 50/50)
• Question 2 (Stochastique): ~14.13  (adversaire optimal)
• Réduction: ~22%  (impact de l'adversaire malveillant)


===============================================================================
🔍 COMMENT LIRE LES FICHIERS DE POLITIQUE
===============================================================================

FICHIER question1_policy.py :
------------------------------
Contient la politique MDP sous forme:
    (grille_hex, pièce): action

Exemple:
    (0x0, 0): 1,    # Grille vide + barre → Action 1
    (0x0, 1): 4,    # Grille vide + angle → Action 4

→ Pour comprendre "Action 1" ou "Action 4", voir actions_legend.txt

FICHIER question2_policy.py :
------------------------------
Contient 3 parties:
1. V = {grille: valeur}           # Valeurs optimales
2. adv_policy = {grille: pièce}   # Choix adversaire (0=barre, 1=angle)
3. player_policy = {grille: {barre:action, angle:action}} # Réponses joueur

Exemple:
    adv_policy = {
        0x0: 1,    # Grille vide → adversaire choisit angle
    }
    
    player_policy = {
        0x0: {
            0: 5,  # Si barre → Action 5
            1: 1,  # Si angle → Action 1
        },
    }

===============================================================================
🧮 NOTATIONS ET CONVENTIONS
===============================================================================

GRILLES (format hexadécimal) :
------------------------------
• 0x0000 = grille vide (16 bits à 0)
• 0xF000 = ligne du haut pleine → ÉTAT TERMINAL
• Chaque chiffre hexa = 4 bits = 1 ligne
• Ex: 0x8c = binaire 10001100 = 
        . . . .
        . . . .
        . . . 1
        . . 1 1

PIÈCES :
--------
• 0 = BARRE (6 actions possibles: 0-5)
• 1 = ANGLE (12 actions possibles: 0-11)

ACTIONS :
---------
• Référence complète dans actions_legend.txt
• Action 4 avec barre ≠ Action 4 avec angle!
• Les positions montrées sont AVANT descente automatique

RÉCOMPENSES :
-------------
• 0 ligne complète: 0 point
• 1 ligne complète: 1 point
• 2 lignes complètes: 3 points
• 3 lignes complètes: 6 points

DISCOUNT(modifiable):
----------
• γ = 0.95 (valeur future décroît de 5% par étape)

===============================================================================
📈 RÉSULTATS ATTENDUS (RÉSUMÉ)
===============================================================================

QUESTION 1:
• États atteignables: 22,162
• Valeur optimale V*(s0): ~18.18
• Meilleure action initiale: Action 4 (angle, rotation 2)
• Temps calcul: ~quelques minutes

QUESTION 2:
• Valeur optimale: ~14.13 (22% moins que Q1)
• Adversaire initial: choisit angle
• Joueur initial: répond Action 1 (angle, rotation 0)
• Simulation 30 steps: score ~11.0

COMPARAISON:
• Réduction théorique: 22.25%
• Réduction pratique: 18.83%
===============================================================================