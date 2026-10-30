from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_adaptyv_egfr import binding_status
from evaluate_adaptyv_egfr import summarize


def test_replicate_status_unknown_is_not_negative():
    assert binding_status([{'binding':'unknown'}])=='unknown'
    assert binding_status([{'binding':'unknown'},{'binding':'False'}])=='false'
    assert binding_status([{'binding':'False'},{'binding':'weak'}])=='true'


def test_missing_affinity_and_boundary_do_not_become_regression_targets():
    def row(i,binding,kd,primary):
        return dict(sample_id=str(i),binding=binding,binding_evaluation=str(binding!='unknown'),
                    quantitative_primary=str(primary),kd_molar=kd,pkd='7' if kd else '',
                    control='False',binder_sequence_cluster80=str(i))
    rows=[row(0,'true','1e-7',True),row(1,'false','',False),row(2,'unknown','',False),row(3,'true','1e-5',False)]
    rows[-1]['pkd']='5'
    m=summarize(rows,{'0':7.,'1':4.,'2':9.,'3':5.})
    assert m['binding_detection']['n']==3 and m['binding_detection']['positives']==2
    assert m['binding_detection']['auroc']==1
    assert m['affinity_primary']['n']==1 and m['affinity_primary']['mae']==0
    assert m['affinity_including_10um_values']['n']==2
    assert m['unknown_binding']==1
    missing=summarize(rows,{'0':7.})
    assert missing['total_designs']==4 and missing['predicted']==1
    assert missing['binding_detection']['auroc'] is None
