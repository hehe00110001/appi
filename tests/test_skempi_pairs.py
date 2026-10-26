import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_skempi_pairs import exact_positive, mutations, partner_key, build


class SkempiPairsTests(unittest.TestCase):
    def fixture(self):
        path = str(Path(__file__).resolve())
        raw = {'#Pdb':'1ABC_A_B', 'Mutation(s)_PDB':'WA10A',
               'Mutation(s)_cleaned':'WA10A', 'Affinity_wt (M)':'1e-9',
               'Affinity_mut (M)':'1e-8', 'Temperature':'298', 'Reference':'123', 'Method':'SPR'}
        feature = {'Source Data Set':'SKEMPI v2.0', 'PDB':'1ABC', 'Mutations':'A_W10A',
                   'Ligand Chains':'B', 'Receptor Chains':'A', 'build_status':'encoded',
                   'KD(M)':'1e-8', 'Affinity PubMed ID':'123', 'Temperature(K)':'298',
                   'split':'train', 'feature_path':path, 'wt_feature_path':path,'csv_line':'2'}
        return raw, feature

    def test_bound_values_are_not_exact_labels(self):
        for value in ['>1e-8', '<1e-9', 'nan', 'inf', '0', '-1']:
            with self.assertRaises(ValueError): exact_positive(value)

    def test_numbering_preserves_insertion_and_sign(self):
        self.assertEqual(mutations('WA-10BA', True), mutations('A_W-10BA'))
        self.assertEqual(partner_key('AB','C'), partner_key('C','BA'))

    def test_pair_sign_and_split_guard(self):
        raw, feature = self.fixture()
        r = build([raw], [feature], [feature])[0]
        self.assertEqual(r['pair_status'], 'paired_pending_leakage_audit')
        self.assertEqual(r['delta_pkd'], -1)
        self.assertAlmostEqual(r['ddg_kcal_mol'], 1.36356, places=4)
        self.assertFalse(r['training_eligible'])
        other = dict(feature, split='test')
        self.assertEqual(build([raw], [feature], [feature, other])[0]['pair_status'], 'split_conflict')

    def test_experiment_and_numbering_not_silently_replaced(self):
        raw, feature = self.fixture()
        self.assertEqual(build([dict(raw, Reference='456')], [feature], [feature])[0]['pair_status'], 'no_feature_match')
        raw['Mutation(s)_cleaned'] = 'WA8A'
        result = build([raw], [feature], [feature])[0]
        self.assertEqual(result['pair_status'], 'paired_pending_leakage_audit')
        self.assertEqual(result['numbering_match'], 'PDB')

    def test_cleaned_numbering_recovers_pair(self):
        raw, feature = self.fixture()
        raw['Mutation(s)_PDB'] = 'WA17A'
        result = build([raw], [feature], [feature])[0]
        self.assertEqual(result['pair_status'], 'paired_pending_leakage_audit')
        self.assertEqual(result['numbering_match'], 'cleaned')

    def test_conflicting_numberings_are_held(self):
        raw, feature = self.fixture()
        raw['Mutation(s)_cleaned'] = 'WA8A'
        other = dict(feature, Mutations='A_W8A', csv_line='3')
        self.assertEqual(build([raw], [feature, other], [feature])[0]['pair_status'], 'ambiguous_feature_match')

    def test_partner_regrouping_is_not_automatic(self):
        raw, feature = self.fixture()
        raw['#Pdb'] = '1ABC_AB_C'
        feature.update({'Ligand Chains':'A', 'Receptor Chains':'B,C'})
        self.assertEqual(build([raw], [feature], [feature])[0]['pair_status'], 'no_feature_match')

    def test_assumed_temperature_and_duplicate(self):
        raw, feature = self.fixture()
        raw['Temperature'] = feature['Temperature(K)'] = '298(assumed)'
        rows = build([raw, raw], [feature], [feature])
        self.assertEqual(rows[0]['pair_status'], 'paired_assumed_temperature')
        self.assertTrue(rows[0]['temperature_assumed'])
        self.assertEqual(rows[1]['duplicate_of'], 2)


if __name__ == '__main__':
    unittest.main()
