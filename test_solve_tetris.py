#test_solve_tetris.py
import unittest
import os
from BabyTetris import BabyTetris
from solve_tetris import build_reachable_mdp, value_iteration, simulate_policy, export_policy_simple

class TestSolveTetris(unittest.TestCase):
    def setUp(self):
        self.env = BabyTetris(discount=0.95)

    def test_build_reachable_mdp(self):
        mdp_data = build_reachable_mdp(self.env, max_states=200000)
        self.assertIn('states', mdp_data)
        self.assertIn('transitions', mdp_data)
        self.assertIn('rewards', mdp_data)
        self.assertIsInstance(mdp_data['states'], list)
        self.assertIsInstance(mdp_data['transitions'], dict)
        self.assertIsInstance(mdp_data['rewards'], dict)
        
        # Vérifier que les transitions sont correctes
        states = mdp_data['states']
        transitions = mdp_data['transitions']
        rewards = mdp_data['rewards']
        
        for s in states:
            if self.env.is_terminal(s):
                continue
            expected_transitions = {}
            for a in self.env.get_actions(s):
                expected_transitions[a] = self.env.get_transitions(s, a)
                self.assertIn((s, a), rewards)
                self.assertEqual(rewards[(s, a)], self.env.get_reward(s, a))
            
            # Vérifier que toutes les transitions sont présentes
            for a, nexts in expected_transitions.items():
                for ns, prob in nexts:
                    self.assertIn((a, ns, prob), transitions[s])

    def test_value_iteration(self):
        mdp_data = build_reachable_mdp(self.env, max_states=1000)
        V, policy = value_iteration(self.env, mdp_data, gamma=0.95, theta=1e-6, max_iters=50)
        self.assertIsInstance(V, dict)
        self.assertIsInstance(policy, dict)

    def test_simulate_policy(self):
        mdp_data = build_reachable_mdp(self.env, max_states=1000)
        V, policy = value_iteration(self.env, mdp_data)
        # Simuler une courte partie
        results = simulate_policy(self.env, policy, episodes=1, max_steps=10, seed=42, render=False)
        self.assertIsInstance(results, list)

    def test_export_policy_simple(self):
        policy = {(0x0000, 0): 0, (0x0000, 1): 1}
        V = {(0x0000, 0): 10.0, (0x0000, 1): 5.0}
        filename = "test_policy.txt"
        export_policy_simple(policy, V, filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)  # Nettoyer

if __name__ == '__main__':
    unittest.main()