import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from resplit_joint_training import (assign_groups, export_rows, group_vectors,
                                   make_groups, validate_export)


def absolute(rid, pdb, feature, wt, split='train'):
    return dict(csv_line=str(rid), PDB=pdb, feature_path=feature, wt_feature_path=wt,
                label_pkd='10', split=split, training_eligible='True',
                **{'Source Data Set': 'example'})


class ResplitTests(unittest.TestCase):
    def fixture(self):
        # Shared WT bridges different PDBs; a third row joins through the second PDB.
        rows = [absolute(1, '1ABC', '/cache/m1.pt', '/cache/shared.pt'),
                absolute(2, '2ABC', '/cache/m2.pt', '/cache/shared.pt', 'val'),
                absolute(3, '2ABC', '/cache/m3.pt', '/cache/w3.pt', 'test'),
                absolute(4, '3ABC', '/cache/m4.pt', '/cache/w4.pt')]
        pair = dict(skempi_csv_line='17', ppb_csv_lines='1',
                    mutant_feature_path='/cache/m1.pt', wt_feature_path='/cache/shared.pt',
                    pkd_mut='10', pkd_wt='11', delta_pkd='-1', split='train',
                    training_eligible='True', **{'#Pdb': '1ABC_A_B'})
        return {'absolute': rows, 'pairs': [pair]}

    def test_transitive_grouping_and_input_order_independence(self):
        data = self.fixture()
        groups = make_groups(data)
        self.assertEqual([g['pdbs'] for g in groups], [['1ABC', '2ABC'], ['3ABC']])
        self.assertEqual(len(groups[0]['records']), 4)
        shuffled = {k: list(reversed(v)) for k, v in data.items()}
        other = make_groups(shuffled)
        self.assertEqual([g['group_id'] for g in groups], [g['group_id'] for g in other])
        np.testing.assert_array_equal(group_vectors(groups)[0], group_vectors(other)[0])

    def test_missing_pair_link_is_rejected(self):
        data = self.fixture()
        data['pairs'][0]['ppb_csv_lines'] = '999'
        with self.assertRaisesRegex(ValueError, 'missing absolute'):
            make_groups(data)

    def test_export_retains_all_fields_and_checks_split_leakage(self):
        data = self.fixture()
        exported = export_rows(make_groups(data), [1, 0])
        audit = validate_export(data, exported, check_files=False)
        self.assertEqual(audit['n_groups'], 2)
        self.assertEqual(len(exported['absolute']), 4)
        self.assertEqual(exported['pairs'][0]['split'], 'val')
        bad = copy.deepcopy(exported)
        bad['absolute'][0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'crosses splits'):
            validate_export(data, bad, check_files=False)
        bad = copy.deepcopy(exported)
        bad['absolute'][0]['label_pkd'] = '9'
        with self.assertRaisesRegex(ValueError, 'field other than'):
            validate_export(data, bad, check_files=False)

    def test_balancing_is_reproducible_and_keeps_whole_groups(self):
        vectors = np.tile([2., 1., 2., 1., 1.], (20, 1))
        args = (vectors, [.8, .1, .1], np.array([2., 2., 1., .25, .1]), 42)
        assignment, counts, targets, score = assign_groups(*args)
        np.testing.assert_array_equal(assignment, assign_groups(*args)[0])
        np.testing.assert_array_equal(counts, targets)
        self.assertEqual(score, 0.)
        self.assertEqual(set(assignment), {0, 1, 2})


if __name__ == '__main__':
    unittest.main()
