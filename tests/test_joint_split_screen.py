import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_joint_splits import alignment_passes


class CoverageTests(unittest.TestCase):
    def test_short_fragment_does_not_clear_external_target(self):
        self.assertTrue(alignment_passes(90,.3,.9,300,100))
        self.assertFalse(alignment_passes(90,.3,.9,300,100,external=True))

    def test_thresholds_and_swapping(self):
        self.assertTrue(alignment_passes(30,.8,.4,100,200))
        self.assertTrue(alignment_passes(30,.4,.8,200,100))
        self.assertFalse(alignment_passes(29.9,1,1,100,100))
        self.assertFalse(alignment_passes(90,.79,.4,100,200))


if __name__ == '__main__': unittest.main()
