"""Input integrity tests for fixed-geometry encoding and partner-group interfaces."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_ppb_features import (interface, apply_mutations, check_sequences, validate_record,
                                DataReview, label_pkd, output_columns, row_base_key)
from ppiformer.utils.bio import amino_acid_to_class
from sequence_mapping import match_partner


class FeaturePreparationTests(unittest.TestCase):
    def test_sequence_list_can_be_permuted_only_within_partner(self):
        self.assertEqual(match_partner(['H','L'], ['AAA','CCC'], ['CCC','AAA']), {'H':1,'L':0})
        with self.assertRaises(ValueError):
            match_partner(['H','L'], ['AAA','AAA'], ['AAA','CCC'])
        row={'Ligand Chains':'H','Receptor Chains':'L',
             'Ligand Sequences':'CCC','Receptor Sequences':'AAA'}
        residues={('H',1,''):{'names':{'ALA'}},('L',1,''):{'names':{'CYS'}}}
        with self.assertRaises(ValueError):
            check_sequences(row,residues,[])

    def test_label_schema_and_consistency(self):
        self.assertEqual(label_pkd({'KD(M)':'1e-9'}), 9.)
        self.assertEqual(label_pkd({'KD(M)':'1e-9','label_pkd':'9'}), 9.)
        with self.assertRaises(DataReview):
            label_pkd({'KD(M)':'1e-9','label_pkd':'8'})
        self.assertEqual(output_columns({'KD(M)':'1e-9'}).count('label_pkd'), 1)
        self.assertEqual(output_columns({'KD(M)':'1e-9','label_pkd':'9'}).count('label_pkd'), 1)

    def test_grouping_canonicalizes_chain_order_and_preserves_rows(self):
        a = {'structure_path':'x','Ligand Chains':'B, A','Receptor Chains':'C','csv_line':'2','split':'train'}
        b = {**a,'Ligand Chains':'A,B','csv_line':'8','split':'test'}
        self.assertEqual(row_base_key(a), row_base_key(b))
        rows = [b.copy(),a.copy()]
        rows.sort(key=row_base_key)
        self.assertEqual({(r['csv_line'],r['split']) for r in rows}, {('2','train'),('8','test')})

    def test_incomplete_nonmutation_residue_is_flagged_not_execution_failure(self):
        row = {'Ligand Chains':'A','Receptor Chains':'B','Mutations':'',
               'Ligand Sequences':'AS','Receptor Sequences':'F'}
        residues = {('A',1,''):{'names':{'ALA'},'atoms':{'CA'}},
                    ('A',2,''):{'names':{'SER'},'atoms':{'N'}},
                    ('B',1,''):{'names':{'PHE'},'atoms':{'CA'}}}
        muts, warnings = validate_record(row, residues)
        self.assertEqual(muts, [])
        self.assertIn('partner_residue_without_CA', warnings)
        with self.assertRaises(DataReview):
            validate_record({**row,'Mutations':'A_S2A'},residues)
        with self.assertRaises(DataReview):
            validate_record({**row,'Ligand Sequences':'WW'},residues)

    def test_mutation_changes_only_type_and_preserves_original(self):
        f = torch.zeros(2, 20)
        f[0, amino_acid_to_class('S')] = 1
        f[1, amino_acid_to_class('F')] = 1
        graph = Data(f=f, y=f.argmax(1), x=torch.randn(2, 3),
                     node_id=['C:SER:16', 'C:PHE:47'])
        muts = [dict(chain='C',residue_number=16,insertion_code='',expected_wt='S',
                     mutant='A',annotation='C_S16A')]
        result = apply_mutations(graph, muts)
        self.assertEqual(graph.node_id[0], 'C:SER:16')
        self.assertEqual(result.node_id[0], 'C:ALA:16')
        self.assertEqual(graph.f[0].argmax().item(), amino_acid_to_class('S'))
        self.assertEqual(result.f[0].argmax().item(), amino_acid_to_class('A'))
        torch.testing.assert_close(graph.x, result.x, atol=0, rtol=0)
        torch.testing.assert_close(graph.f[1], result.f[1], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'outside_encoded_interface'):
            apply_mutations(graph, [{**muts[0], 'residue_number': 99}])
        with self.assertRaisesRegex(ValueError, 'wt_type_mismatch'):
            apply_mutations(graph, [{**muts[0], 'expected_wt': 'W'}])

    def test_interface_ignores_contacts_within_partner(self):
        def atom(i, chain, x):
            return f'ATOM  {i:5d}  CA  ALA {chain}{1:4d}    {x:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{20.:6.2f}           C  \n'
        with tempfile.TemporaryDirectory() as temp:
            source, out = Path(temp)/'source.pdb', Path(temp)/'interface.pdb'
            source.write_text(atom(1,'A',0) + atom(2,'B',1) + atom(3,'C',100) + atom(4,'D',101))
            with self.assertRaisesRegex(ValueError, 'no_cross_partner_interface'):
                interface(source, ['A','B'], ['C','D'], 10, out)
            source.write_text(atom(1,'A',0) + atom(2,'B',1) + atom(3,'C',9) + atom(4,'D',101))
            selected = interface(source, ['A','B'], ['C','D'], 10, out)
            self.assertEqual(selected, {('A',1,''),('B',1,''),('C',1,'')})

    def test_sequence_check_applies_mutation_and_allows_missing_coordinates(self):
        row = {'Ligand Chains':'A','Receptor Chains':'B',
               'Ligand Sequences':'GAV','Receptor Sequences':'F'}
        residues = {('A',2,''):{'names':{'SER'}},('B',1,''):{'names':{'PHE'}}}
        mutations = [dict(chain='A',residue_number=2,insertion_code='',mutant='A')]
        check_sequences(row, residues, mutations)
        with self.assertRaisesRegex(ValueError, 'sequence_not_in_csv'):
            check_sequences(row, residues, [])


if __name__ == '__main__':
    unittest.main()
