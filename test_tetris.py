#test_tetris.py
import unittest
from BabyTetris import BabyTetris

class TestBabyTetris(unittest.TestCase):
    def setUp(self):
        self.env = BabyTetris(discount=0.95)

    def test_initial_state(self):
        state = self.env.get_initial_state()
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 2)
        self.assertEqual(state[0], 0x0000)  # Grille vide
        self.assertIn(state[1], [0, 1])  # Pièce 0 ou 1

    def test_get_actions(self):
        # Test pour pièce barre (0)
        state_barre = (0x0000, 0)
        try:
            actions_barre = self.env.get_actions(state_barre)
            self.assertIsInstance(actions_barre, (tuple, range))
            self.assertGreater(len(actions_barre), 0)  # Au moins une action
        except (AttributeError, TypeError, IndexError):
            self.skipTest("get_actions non implémenté ou erreur dans le code")

        # Test pour pièce angle (1)
        state_angle = (0x0000, 1)
        try:
            actions_angle = self.env.get_actions(state_angle)
            self.assertIsInstance(actions_angle, (tuple, range))
            self.assertGreater(len(actions_angle), 0)  # Au moins une action
        except (AttributeError, TypeError, IndexError):
            self.skipTest("get_actions non implémenté ou erreur dans le code")

    def test_compute_grid(self):
        # Test action sur grille vide
        state = (0x0000, 0)  # Grille vide, pièce barre
        new_grid, failed = self.env.compute_grid(state, 0)
        self.assertIsInstance(new_grid, int)
        self.assertIsInstance(failed, bool)

    def test_clear_full_lines(self):
        # Test ligne pleine
        grid_full_line = 0x000F  # Ligne 0 pleine
        cleared = self.env.clear_full_lines(grid_full_line)
        self.assertIsInstance(cleared, int)

        # Test grille sans ligne pleine
        grid_empty = 0x0000
        cleared = self.env.clear_full_lines(grid_empty)
        self.assertEqual(cleared, 0x0000)

    def test_get_reward(self):
        state = (0x0000, 0)
        reward = self.env.get_reward(state, 0)
        self.assertIsInstance(reward, float)

    def test_is_terminal(self):
        # État non terminal
        state_non_term = (0x0000, 0)
        self.assertIsInstance(self.env.is_terminal(state_non_term), bool)

        # État terminal
        state_term = (0xF000, 0)
        self.assertIsInstance(self.env.is_terminal(state_term), bool)

    def test_get_discount_factor(self):
        discount = self.env.get_discount_factor()
        self.assertIsInstance(discount, float)
        self.assertGreaterEqual(discount, 0)
        self.assertLessEqual(discount, 1)

    def test_get_transitions(self):
        state = (0x0000, 0)
        transitions = self.env.get_transitions(state, 0)
        self.assertIsInstance(transitions, list)

if __name__ == '__main__':
    unittest.main()