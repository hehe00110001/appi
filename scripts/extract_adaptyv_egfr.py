"""Frozen PPIformer and full-chain ESM features for externally held-out EGFR designs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import torch
import esm
import hydra
from omegaconf import OmegaConf
from torch_geometric.data import Batch
from ppiformer.model.ppiformer import PPIformer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from appi.structure_context_data import file_digest, StructureContextData
from build_ppb_features import interface, node_key
from train_affinity import atomic_save, json_save
from extract_esm2_features import window_starts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'data/processed/adaptyv_egfr_features_v1')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'extract.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=ROOT/'data/processed/adaptyv_egfr_v1'
    m=json.loads((source/'structure_manifest.json').read_text())
    selected=dict(sorted(m['graphs'].items()))
    if args.limit:
        # Cover both structural pipelines in the engineering smoke test.
        selected={k:v for prefix in ('r1_','r2_') for k,v in list((k,v) for k,v in selected.items() if k.startswith(prefix))[:args.limit]}
    old=json.loads((ROOT/'data/processed/esm2_complex40_v1/extraction_config.json').read_text())
    preprocessing=json.loads((ROOT/'data/processed/ppb_ppiformer/config.json').read_text())
    ppi_checkpoint=ROOT/'external/PPIformer/weights/masked_modeling.ckpt'
    esm_checkpoint=Path('/home/weizg/wei/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')
    regression=esm_checkpoint.with_name('esm2_t33_650M_UR50D-contact-regression.pt')
    assert file_digest(ppi_checkpoint)==preprocessing['checkpoint_sha256']
    assert file_digest(esm_checkpoint)==old['checkpoint_sha256'] and file_digest(regression)==old['regression_sha256']
    config=dict(source_manifest_sha256=file_digest(source/'structure_manifest.json'),records_sha256=file_digest(source/'audited_records.csv'),
                ppi_checkpoint_sha256=preprocessing['checkpoint_sha256'],esm_checkpoint_sha256=old['checkpoint_sha256'],
                script_sha256=file_digest(Path(__file__)),interface_code_sha256=file_digest(ROOT/'scripts/build_ppb_features.py'),
                radius_angstrom=preprocessing['radius_angstrom'],ppi_device='cpu',esm_device=args.device,limit=args.limit)
    cp=args.out/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config,'Cache configuration changed'
    else:json_save(config,cp)
    # PPIformer uses CPU exactly as the training-feature writer.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    try:encoder=PPIformer.load_from_checkpoint(ppi_checkpoint,map_location='cpu').eval().requires_grad_(False)
    finally:os.environ.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD',None)
    cfg=OmegaConf.create(encoder.hparams['run_cfg'])
    transforms=[t for t in cfg.val_dataloader.pretransform if t['_target_']!='ppiformer.data.transforms.DDGLabelPretransform']
    assert len(transforms)==1
    transform=hydra.utils.instantiate(transforms[0])
    graph_dir=args.out/'graphs';graph_dir.mkdir(exist_ok=True)
    graphs,records,failures={},{},{}
    for number,(name,g) in enumerate(selected.items(),1):
        path=Path(g['structure_path']);assert file_digest(path)==g['structure_sha256']
        cached=graph_dir/(name+'.pt')
        if cached.exists():
            save=torch.load(cached,map_location='cpu',weights_only=True)
        else:
            ligand=[c for c,v in g['chains'].items() if v['partner']];receptor=[c for c,v in g['chains'].items() if not v['partner']]
            crop=args.out/'interfaces'/(name+'.pdb')
            try:chosen=interface(path,ligand,receptor,config['radius_angstrom'],crop)
            except ValueError as error:
                if str(error)!='no_cross_partner_interface':raise
                failures[name]=str(error);continue
            graph=transform(crop);ids=list(graph.node_id)
            assert {node_key(n) for n in ids}==chosen and len(ids)==len(chosen)
            partner=torch.tensor([n.split(':')[0] in ligand for n in ids],dtype=torch.bool)
            assert partner.any() and (~partner).any()
            with torch.inference_mode():h=encoder(Batch.from_data_list([graph])).detach().cpu().clone()
            assert h.shape==(len(ids),128) and torch.isfinite(h).all()
            nodes=[]
            for node in ids:
                chain,num,insert=node_key(node);nodes.append([chain,g['positions'][f'{chain}:{num}:{insert}']])
            save=dict(graph=dict(h=h,coordinates=graph.x.cpu().clone(),partner=partner,node_id=ids),nodes=nodes)
            atomic_save(save,cached)
        graphs[name]=save['graph'];records[name]=dict(chains=g['chains'],nodes=save['nodes'],structure_path=str(path),structure_sha256=g['structure_sha256'])
        if number%50==0 or number==1:print(f'PPIformer {number}/{len(selected)}',flush=True)
    del encoder
    sequence_ids={c['sequence_id'] for g in records.values() for c in g['chains'].values()}
    sequences={k:m['sequences'][k] for k in sorted(sequence_ids)}
    manifest=dict(metadata=dict(external_only=True,source_sha256=config['source_manifest_sha256']),graphs=records,sequences=sequences)
    json_save(manifest,args.out/'manifest.json');json_save(failures,args.out/'failures.json')
    extraction=dict(old,manifest_sha256=file_digest(args.out/'manifest.json'),script_sha256=file_digest(Path(__file__)))
    json_save(extraction,args.out/'extraction_config.json')
    cache=args.out/'sequences';cache.mkdir(exist_ok=True)
    model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',
        torch.load(esm_checkpoint,map_location='cpu',weights_only=False),torch.load(regression,map_location='cpu',weights_only=True))
    model.eval().requires_grad_(False).to(args.device);converter=alphabet.get_batch_converter()
    for number,(key,seq) in enumerate(sequences.items(),1):
        path=cache/(key+'.pt')
        if not path.exists():
            result=torch.zeros(len(seq),1280);count=torch.zeros(len(seq),1)
            for start in window_starts(len(seq)):
                subseq=seq[start:start+1022];_,_,tokens=converter([(key,subseq)])
                with torch.inference_mode():h=model(tokens.to(args.device),repr_layers=[33],return_contacts=False)['representations'][33]
                result[start:start+len(subseq)]+=h[0,1:len(subseq)+1].cpu().float();count[start:start+len(subseq)]+=1
            assert (count>0).all();result/=count;assert torch.isfinite(result).all()
            atomic_save(dict(sequence=seq,h=result.half()),path)
        if number%50==0 or number==1:print(f'ESM {number}/{len(sequences)}',flush=True)
    del model
    atomic_save(graphs,args.out/'residue_features.pt')
    verified=StructureContextData.load(graphs,args.out)
    json_save(dict(complete=True,smoke_test=bool(args.limit),graphs=len(graphs),sequences=len(sequences),failures=failures,
                   structure_cache_sha256=file_digest(args.out/'residue_features.pt'),sequence_set_sha256=verified.provenance['sequence_set_sha256']),args.out/'complete.json')
    print('Completed feature extraction',flush=True)

if __name__=='__main__':main()
