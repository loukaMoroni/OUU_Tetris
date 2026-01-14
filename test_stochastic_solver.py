#test_stochastic_solver.py
import unittest
from stochastic_solver import StochasticSolver
from BabyTetris import BabyTetris

class TestStochasticSolver(unittest.TestCase):
    def setUp(self):
        self.env = BabyTetris(discount=0.95)
        self.solver = StochasticSolver(discount=0.95)

    def test_extract_grids_from_mdp(self):
        mdp_data = {'states': [(0x0000, 0)]}
        grids = self.solver.extract_grids_from_mdp(mdp_data)
        self.assertIsInstance(grids, list)
        self.assertGreater(len(grids), 0)

    def test_build_transitions_from_mdp(self):
        mdp_data = {'transitions': {}, 'states': [(0x0000, 0)]}
        transitions = self.solver.build_transitions_from_mdp(mdp_data)
        self.assertIsInstance(transitions, dict)

    def test_solve_stochastic_game(self):
        mdp_data = {'states': [(0x0000, 0)], 'transitions': {}, 'rewards': {}}
        V, adversary_policy, player_policy, transitions = self.solver.solve_stochastic_game(mdp_data, theta=1e-6, max_iters=10)
        self.assertIsInstance(V, dict)
        self.assertIsInstance(adversary_policy, dict)
        self.assertIsInstance(player_policy, dict)
        self.assertIsInstance(transitions, dict)

    def test_extract_policies(self):
        V = {0x0000: 10.0}
        actions_map = {(0x0000, 0): [(0, (0x0001, 1.0))]}
        adversary_policy, player_policy = self.solver._extract_policies(V, actions_map)
        self.assertIsInstance(adversary_policy, dict)
        self.assertIsInstance(player_policy, dict)

    def test_is_terminal_grid(self):
        grid = 0x0000
        self.assertFalse(self.solver.is_terminal_grid(grid))
        grid_term = 0xF000
        self.assertTrue(self.solver.is_terminal_grid(grid_term))

if __name__ == '__main__':
    unittest.main()