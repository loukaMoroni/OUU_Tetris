#solve_tetris.py
import random
import collections
import math
from BabyTetris import BabyTetris

# ---------------------------------------------------------
#BFS (Breadth-First Search)
# ---------------------------------------------------------
def build_reachable_mdp(env: BabyTetris, max_states: int = 200000):
    """
    BFS exploration from the initial state to collect:
        - the reachable states
        - the transition model
        - the reward model
    """
    start = env.get_initial_state()
    # queue containing states to explore
    q = collections.deque([start])
    # States already visited
    visited = {start}
    transitions = {}
    rewards = {}

    while q:
        # the state to explore is at the front of the queue and removed from it
        s = q.popleft()
        transitions[s] = []
        # for each action possible from this state ,we get the reward and the transitions
        for a in env.get_actions(s):
            r = env.get_reward(s, a)
            rewards[(s, a)] = r
            nexts = env.get_transitions(s, a)
            # for each possible next state and its probability from state s with action a
            for (ns, prob) in nexts:
                # we store the transition
                transitions[s].append((a, ns, prob))
                # if the next state has not been visited yet , we add it to the queue and to visited
                if ns not in visited and len(visited) < max_states:
                    visited.add(ns)
                    q.append(ns)
        # safety check to avoid exploding state spaces
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
    Value iteration on reachable set.
    Return V (dict state->value) and policy (dict state->best_action_index or None).
    """
    if gamma is None:
        gamma = env.get_discount_factor()

    states = mdp_data['states']
    transitions = mdp_data['transitions']
    rewards = mdp_data['rewards']
    # initialize V(s)=0 for all states
    V = {s: 0.0 for s in states}
    for it in range(max_iters):
        delta = 0.0
        for s in states:
            if env.is_terminal(s):
                
                v_new = 0.0
            else:
                best_val = -math.inf #best action value
                # for each action possible from state s
                for a in env.get_actions(s):
                    r = rewards.get((s, a), 0.0)
                    exp = 0.0
                    #for each possible next state and its probability from state s with action a
                    for (act, ns, prob) in transitions.get(s, []):
                        if act != a:
                            continue
                        exp += prob * V.get(ns, 0.0) 
                    #bellman update
                    val = r + gamma * exp
                    if val > best_val:
                        best_val = val
                v_new = best_val if best_val != -math.inf else 0.0
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        # stopping condition
        if delta < theta:
            print(f"Value iteration converged at iter {it} (delta={delta})")
            break

    #extract deterministic policy
    policy = {}

    for s in states:
        if env.is_terminal(s):
            policy[s] = None
            continue
        best_a = None
        best_val = -math.inf
        #compute value for each action using current V
        for a in env.get_actions(s):
            r = rewards.get((s, a), 0.0)
            exp = 0.0
            #Expected value over next states
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
    Simulate the given policy for a number of episodes.
    Return the list of returns (one per episode).
    """
    # set random seed for reproducibility simulations
    if seed is not None:
        random.seed(seed)

    gamma = env.get_discount_factor()
    returns = []

    for ep in range(episodes):
        state = env.get_initial_state()
        env.state = state  
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
            #update discounted G
            G += (gamma ** t) * reward

            #save trajectory for rendering
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
# Usage example
# -------------------------
if __name__ == "__main__":
    env = BabyTetris(discount=0.95)
    print("Initial state:", env.get_initial_state())

    print("Exploration des états atteignables (BFS)...")
    mdp = build_reachable_mdp(env, max_states=200000)
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

# Disable rendering to prevent console buffer overflow and read the initial stats (BFS count, V_init)
rets = simulate_policy(env, policy, episodes=5, max_steps=200, seed=42, render=False) 