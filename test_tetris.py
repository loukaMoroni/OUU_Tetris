from BabyTetris import BabyTetris

# Create an instance of BabyTetris
env = BabyTetris(discount=0.99)
print("Instance créée :", env)

# Get the initial state
print("État initial =", env.get_initial_state())
print("État courant =", env.state)

# Get possible actions from the initial state
state = env.state
actions = list(env.get_actions(state))
print("Actions possibles :", actions)

# Compute the grid after taking action 0
new_grid, failed = env.compute_grid(env.state, 0)
print(f"Grille après action 0 = {hex(new_grid)}")
print("Collision ?", failed)

# Function to display the grid in a readable format
def show_grid(grid):
    bits = f"{grid:016b}"
    print(bits[0:4])
    print(bits[4:8])
    print(bits[8:12])
    print(bits[12:16])

print("Grille binaire :")
show_grid(new_grid)

# Get transitions from the current state with action 0
transitions = env.get_transitions(env.state, 0)
print("Transitions =", transitions)

# Test reward
reward = env.get_reward(env.state, 0)
print("Récompense pour action 0 =", reward)
