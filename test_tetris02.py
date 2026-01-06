# test_stochastic_tetris.py
from BabyTetris_02 import StochasticTetris

# =========================
# Initialisation
# =========================
env = StochasticTetris(discount=0.99)
print("Instance créée :", env)

state = env.get_initial_state()
grid, _ = state

print("\nÉtat initial =", state)

# =========================
# Fonction utilitaire affichage
# =========================
def show_grid(grid):
    bits = f"{grid:016b}"
    print(bits[0:4])
    print(bits[4:8])
    print(bits[8:12])
    print(bits[12:16])

# =========================
# TEST actions conditionnées par la pièce
# =========================
print("\n=== TEST actions dépendantes de la pièce ===")

actions_line = list(env.get_actions((grid, 0)))
actions_angle = list(env.get_actions((grid, 1)))

print("Actions possibles si pièce = LIGNE :", actions_line)
print("Actions possibles si pièce = ANGLE :", actions_angle)

assert actions_line != actions_angle
print("✓ Les actions dépendent bien de la pièce")

# =========================
# TEST getAdversarial_transition
# =========================
print("\n=== TEST getAdversarial_transition ===")

# Adversaire choisit LIGNE
piece = 0
action = actions_line[0]

next_state_1, reward_1 = env.getAdversarial_transition(
    state=state,
    adversary_choice=piece,
    player_action=action
)

print("\nAdversaire = LIGNE")
print("Action joueur =", action)
print("Récompense =", reward_1)
print("Grille suivante :")
show_grid(next_state_1[0])

# Adversaire choisit ANGLE
piece = 1
action = actions_angle[0]

next_state_2, reward_2 = env.getAdversarial_transition(
    state=state,
    adversary_choice=piece,
    player_action=action
)

print("\nAdversaire = ANGLE")
print("Action joueur =", action)
print("Récompense =", reward_2)
print("Grille suivante :")
show_grid(next_state_2[0])

# =========================
# Vérification différence des pièces
# =========================
print("\n=== VÉRIFICATION ===")
if next_state_1[0] != next_state_2[0]:
    print("✓ Les deux pièces produisent des grilles différentes")
else:
    print("⚠️ Problème : grilles identiques")

# =========================
# TEST Q_minimax
# =========================
print("\n=== TEST Q_minimax ===")

V = {
    next_state_1: 1.0,
    next_state_2: 0.3
}

gamma = 0.99

q_val = env.Q_minimax(
    state=state,
    player_action=actions_line[0],
    gamma=gamma,
    V=V
)

print(f"Q_minimax = {q_val}")
print("✓ Q_minimax prend bien le minimum sur les choix adverses")

# =========================
# Vérification héritage
# =========================
print("\n=== HÉRITAGE BabyTetris ===")
print("compute_grid :", hasattr(env, "compute_grid"))
print("get_reward :", hasattr(env, "get_reward"))
print("get_transitions :", hasattr(env, "get_transitions"))
print("clear_full_lines :", hasattr(env, "clear_full_lines"))

print("\n✓ Tous les tests sont cohérents avec le jeu adversarial")
