import unittest

from appi.units import pkd_from_molar, delta_pkd_to_ddg, ddg_to_delta_pkd, energy_factor


class UnitsTests(unittest.TestCase):
    def test_tenfold_stronger_binding(self):
        delta = pkd_from_molar(1e-9) - pkd_from_molar(1e-8)
        self.assertAlmostEqual(delta, 1.0)
        self.assertAlmostEqual(delta_pkd_to_ddg(delta), -1.364247, places=5)
        self.assertAlmostEqual(ddg_to_delta_pkd(delta_pkd_to_ddg(delta)), delta)

    def test_invalid_measurements(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                pkd_from_molar(value)
            with self.assertRaises(ValueError):
                energy_factor(value)

