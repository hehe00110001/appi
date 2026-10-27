import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from split_complex_homology40 import Union, combined_sequence, assign_preserving


class ComplexHomologySplitTests(unittest.TestCase):
    def test_author_concatenation_keeps_chain_order(self):
        self.assertEqual(combined_sequence({'Ligand Sequences':'AC,DE','Receptor Sequences':'FG,HI'}),'ACDEFGHI')
        with self.assertRaises(ValueError):
            combined_sequence({'Ligand Sequences':'AC*','Receptor Sequences':'DE'})

    def test_transitive_grouping(self):
        u=Union(5);u.join(0,1);u.join(2,3);u.join(1,3)
        self.assertEqual(len({u.find(i) for i in range(4)}),1)
        self.assertNotEqual(u.find(0),u.find(4))

    def test_count_balance_determinism_and_no_label_dependence(self):
        groups=[]
        for i in range(30):
            split='train' if i<25 else 'val' if i<28 else 'test'
            records=[('absolute',{'split':split,'Source Data Set':'SKEMPI','label_pkd':str(i)}) for _ in range(3)]
            records.append(('pairs',{'split':split}))
            groups.append({'records':records})
        a,counts,_,_=assign_preserving(groups,[.8,.1,.1])
        for g in groups:
            for kind,row in g['records']:
                if kind=='absolute':row['label_pkd']='999999'
        b,counts2,_,_=assign_preserving(groups,[.8,.1,.1])
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(counts,counts2)
        np.testing.assert_array_equal(counts[:,:2].sum(0),[90,30])
        self.assertTrue((counts[:,:2]>0).all())


if __name__=='__main__':unittest.main()
