#test_analyser.py
import unittest
from analyzer import PolicyAnalyzer

class TestPolicyAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = PolicyAnalyzer(discount=0.95)

    def test_is_terminal_grid(self):
        grid = 0x0000
        self.assertFalse(self.analyzer.is_terminal_grid(grid))
        grid_term = 0xF000
        self.assertTrue(self.analyzer.is_terminal_grid(grid_term))

    def test_print_policy_summary(self):
        V = {0x0000: 10.0}
        adversary_policy = {0x0000: 0}  # grid -> piece
        player_policy = {0x0000: {0: 0, 1: 1}}  # grid -> {piece: action}
        # Cette méthode imprime, vérifier qu'elle ne crash pas
        self.analyzer.print_policy_summary(V, adversary_policy, player_policy)
        self.assertTrue(True)  # Placeholder

    def test_analyze_adversary_choices(self):
        adversary_policy = {0x0000: 0, 0x0001: 1}
        self.analyzer.analyze_adversary_choices(adversary_policy)
        self.assertTrue(True)  # Placeholder

    def test_analyze_values(self):
        V = {0x0000: 10.0, 0x0001: 5.0, 0xF000: 0.0}
        self.analyzer.analyze_values(V)
        self.assertTrue(True)  # Placeholder

    def test_compare_values(self):
        mdp_value = 10.0
        stochastic_value = 9.0
        reduction = self.analyzer.compare_values(mdp_value, stochastic_value)
        self.assertIsInstance(reduction, float)

    def test_export_stochastic_policies_simple(self):
        V = {0x0000: 10.0}
        adv_policy = {0x0000: 0}
        player_policy = {0x0000: {0: 0}}
        filename = "test_policy_q2.txt"
        result = self.analyzer.export_stochastic_policies_simple(V, adv_policy, player_policy, filename)
        self.assertEqual(result, filename)
        import os
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_print_conclusion(self):
        reduction = 5.0
        mdp_reward = 10.0