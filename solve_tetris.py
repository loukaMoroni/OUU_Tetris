import random
import collections
import math
from BabyTetris import BabyTetris

# ---------------------------------------------------------
# NOTE ABOUT BFS (Breadth-First Search)
# ---------------------------------------------------------
# This BFS is NOT part of the MDP theory taught in class.
# It is NOT required by the course and it is NOT a standard 
# component of Value Iteration or Policy Iteration.
#
# Why do we use it here?
# ----------------------
# In Baby Tetris, the total theoretical state space is:
#     4x4 grid = 16 bits = 65 536 possible grids
#     each combined with 2 possible next pieces
#     → 131 072 total states
#
# However, most of these states are impossible to reach 
# by actually playing the game (blocked patterns, illegal shapes, 
# unreachable configurations, etc.).
#
# BFS is therefore used ONLY as a PRACTICAL optimization:
#    - it starts from the initial state,
#    - explores only states that can actually be reached by playing,
#    - and ignores the huge amount of unreachable states.
#
# This makes Value Iteration MUCH faster in practice.
# Again: this is a practical engineering trick, 
# NOT part of the theoretical definition of an MDP.
#
# If needed, the MDP can be solved WITHOUT BFS 
# by running Value Iteration on all 131 072 states.
# ---------------------------------------------------------
def build_reachable_mdp(env: BabyTetris, max_states: int = 200000):
    """
    BFS exploration from the initial state to collect:
        - the reachable states
        - the transition model
        - the reward model
    """
    start = env.get_initial_state()
    q = collections.deque([start])
    visited = {start}
    transitions = {}
    rewards = {}

    while q:
        s = q.popleft()
        transitions[s] = []
        for a in env.get_actions(s):
            r = env.get_reward(s, a)
            rewards[(s, a)] = r
            nexts = env.get_transitions(s, a)
            for (ns, prob) in nexts:
                transitions[s].append((a, ns, prob))
                if ns not in visited and len(visited) < max_states:
                    visited.add(ns)
                    q.append(ns)
        if len(visited) >= max_states:
            print("WARNING: max_states reached in build_reachable_mdp")
            break

    return {
        'states': list(visited),
        'transitions': transitions,
        'rewards': rewards
    }


# -------------------------
# Value Iteration
# -------------------------
def value_iteration(env: BabyTetris, mdp_data, gamma=None, theta=1e-8, max_iters=10000):
    """
    Value iteration sur l'ensemble reachable.
    Retourne V (dict state->value) et policy (dict state->best_action_index or None).
    """
    if gamma is None:
        gamma = env.get_discount_factor()

    states = mdp_data['states']
    transitions = mdp_data['transitions']
    rewards = mdp_data['rewards']

    V = {s: 0.0 for s in states}

    for it in range(max_iters):
        delta = 0.0
        for s in states:
            if env.is_terminal(s):
                v_new = 0.0
            else:
                best_val = -math.inf
                for a in env.get_actions(s):
                    r = rewards.get((s, a), 0.0)
                    exp = 0.0
                    # sum over transitions for this state-action
                    for (act, ns, prob) in transitions.get(s, []):
                        if act != a:
                            continue
                        exp += prob * V.get(ns, 0.0) 
                        # ----------------------
                    val = r + gamma * exp
                    if val > best_val:
                        best_val = val
                v_new = best_val if best_val != -math.inf else 0.0
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        # critère d'arrêt
        if delta < theta:
            print(f"Value iteration converged at iter {it} (delta={delta})")
            break

    # extraction de politique déterministe
    policy = {}
    for s in states:
        if env.is_terminal(s):
            policy[s] = None
            continue
        best_a = None
        best_val = -math.inf
        for a in env.get_actions(s):
            r = rewards.get((s, a), 0.0)
            exp = 0.0
            for (act, ns, prob) in transitions.get(s, []):
                if act != a:
                    continue
                exp += prob * V.get(ns, 0.0)
            val = r + gamma * exp
            if val > best_val:
                best_val = val
                best_a = a
        policy[s] = best_a
    return V, policy

# -------------------------
# Simulate policy
# -------------------------
def simulate_policy(env: BabyTetris, policy, episodes=10, max_steps=200, seed=None, render=False):
    """
    Simule la politique donnée `policy` sur `episodes` épisodes.
    Retourne la liste des retours discountés pour chaque épisode.
    Si policy[state] is None => episode terminal, on arrête.
    """
    if seed is not None:
        random.seed(seed)

    gamma = env.get_discount_factor()
    returns = []

    for ep in range(episodes):
        state = env.get_initial_state()
        env.state = state  # synchronise l'env si nécessaire
        G = 0.0
        t = 0
        trajectory = []
        while t < max_steps and not env.is_terminal(state):
            a = policy.get(state)
            if a is None:
                break
            # safety: if policy has an action not in get_actions, fallback random
            if a not in env.get_actions(state):
                a = random.choice(list(env.get_actions(state)))

            # sample next state according to transition probabilities
            nexts = env.get_transitions(state, a)
            # nexts is list of (ns, prob). Sample:
            r = random.random()
            cum = 0.0
            chosen_ns = None
            for (ns, prob) in nexts:
                cum += prob
                if r <= cum:
                    chosen_ns = ns
                    break
            if chosen_ns is None:
                chosen_ns = nexts[-1][0]  # fallback numerical

            reward = env.get_reward(state, a)
            G += (gamma ** t) * reward

            if render:
                trajectory.append((state, a, reward, chosen_ns))

            state = chosen_ns
            env.state = state
            t += 1

        returns.append(G)
        if render:
            print(f"Episode {ep+1} (steps {t}) G={G}")
            for s,a,r,ns in trajectory:
                print("  ", s, "a=", a, "r=", r, "->", ns)
            print("-"*40)
    return returns

# -------------------------
# Exemple d'utilisation
# -------------------------
if __name__ == "__main__":
    env = BabyTetris(discount=0.95)
    print("Initial state:", env.get_initial_state())

    print("Exploration des états atteignables (BFS)...")
    mdp = build_reachable_mdp(env, max_states=20000)
    print("Nombre d'états atteignables:", len(mdp['states']))

    print("Lancement de la value iteration...")
    V, policy = value_iteration(env, mdp, gamma=env.get_discount_factor(), theta=1e-7)
    init = env.get_initial_state()
    print("Valeur de l'état initial :", V.get(init))
    print("Meilleure action pour l'état initial :", policy.get(init))

    # Simuler la politique
    print("\nSimulation de la politique optimale (5 épisodes)...")
    rets = simulate_policy(env, policy, episodes=5, max_steps=200, seed=42, render=True)
    print("Retours par épisode :", rets)
    print("Moyenne retour :", sum(rets)/len(rets))