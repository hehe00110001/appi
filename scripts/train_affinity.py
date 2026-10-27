"""Frozen PPIformer features: absolute pKd baseline or shared-head difference loss."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import fcntl

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.model import CachedAffinityRegressor


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_save(value, path):
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def json_save(value, path):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp,path)


def metrics(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if not len(pred): return {'n':0}
    error = pred-target
    varying = len(pred)>1 and np.std(pred)>0 and np.std(target)>0
    return {'n':len(pred),'rmse':float(np.sqrt(np.mean(error**2))),
            'mae':float(np.mean(abs(error))),
            'pearson':float(np.corrcoef(pred,target)[0,1]) if varying else None,
            'spearman':float(spearmanr(pred,target).statistic) if varying else None}


def read_tables(directory, limit):
    tables = {}
    feature_splits = {}
    for split in ['train','val','test']:
        for kind in ['absolute','pairs']:
            with (directory/f'{kind}_{split}.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            if not rows: raise ValueError(f'Empty {kind}/{split}')
            for r in rows:
                if r['split']!=split or r['training_eligible']!='True':
                    raise ValueError('Ineligible or incorrectly split record')
                for key in ['feature_path','wt_feature_path','mutant_feature_path']:
                    if r.get(key):
                        if feature_splits.setdefault(r[key],split)!=split:
                            raise ValueError('Feature shared across splits: '+r[key])
            tables[kind,split] = rows[:limit] if limit else rows
    return tables


def load_features(tables):
    paths = sorted({r[key] for rows in tables.values() for r in rows
                    for key in ['feature_path','wt_feature_path','mutant_feature_path'] if r.get(key)})
    features = {}
    feature_digest = hashlib.sha256()
    for i,path in enumerate(paths,1):
        # These local cache files are produced by this project's feature writer.
        saved = torch.load(path, map_location='cpu', weights_only=True)
        z = saved['pooled'].float()
        if z.shape!=(768,) or not torch.isfinite(z).all(): raise ValueError('Invalid pooled feature: '+path)
        features[path] = z
        feature_digest.update(path.encode());feature_digest.update(z.numpy().tobytes())
        if i%500==0: print(f'Loaded {i}/{len(paths)} feature files',flush=True)
    return features, feature_digest.hexdigest()


def make_tensors(rows, features, kind):
    if kind=='absolute':
        return (torch.stack([features[r['feature_path']] for r in rows]),
                torch.tensor([float(r['label_pkd']) for r in rows]))
    return (torch.stack([features[r['mutant_feature_path']] for r in rows]),
            torch.stack([features[r['wt_feature_path']] for r in rows]),
            torch.tensor([float(r['delta_pkd']) for r in rows]))


def pair_endpoint_targets(rows):
    targets = torch.tensor([[float(r['pkd_mut']), float(r['pkd_wt'])] for r in rows])
    delta = torch.tensor([float(r['delta_pkd']) for r in rows])
    if not torch.isfinite(targets).all() or not torch.allclose(
            targets[:, 0]-targets[:, 1], delta, rtol=1e-5, atol=2e-6):
        raise ValueError('Pair endpoint labels are invalid or inconsistent with delta_pkd')
    return targets


def pair_losses(mut_pred, wt_pred, endpoints, delta, scale):
    """Mean absolute endpoint loss and difference loss on the same experiment batch."""
    endpoint_loss = .5 * (((mut_pred-endpoints[:, 0])/scale).square()
                          + ((wt_pred-endpoints[:, 1])/scale).square()).mean()
    difference_loss = ((mut_pred-wt_pred-delta)/scale).square().mean()
    return endpoint_loss, difference_loss


@torch.no_grad()
def evaluate(model, tensors, device, batch_size):
    model.eval()
    predictions = []
    for start in range(0,len(tensors[-1]),batch_size):
        batch = [x[start:start+batch_size].to(device) for x in tensors]
        pred = model(batch[0]) if len(batch)==2 else model.delta(batch[0],batch[1])
        predictions.extend(pred.cpu().tolist())
    return metrics(predictions,tensors[-1].tolist()), predictions


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/processed/joint_training_v2')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['absolute','joint'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--patience',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--lambda-diff',type=float,default=1.)
    p.add_argument('--pair-absolute-weight',type=float,default=0.,
                   help='Weight of mean WT/mutant absolute endpoint loss in BOTH modes; '
                        'use 1 for the matched-label resplit comparison, 0 for the legacy objective.')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--limit',type=int,default=0,help='Smoke test only; limit each table')
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    if (args.epochs<1 or args.batch_size<1 or args.patience<1
            or not math.isfinite(args.lambda_diff) or args.lambda_diff<0
            or not math.isfinite(args.pair_absolute_weight) or args.pair_absolute_weight<0):
        p.error('Invalid training parameters')
    args.out.mkdir(parents=True,exist_ok=True)
    lock = (args.out/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tables = read_tables(args.data,args.limit)
    features, feature_hash = load_features(tables)
    tensors = {key:make_tensors(rows,features,key[0]) for key,rows in tables.items()}
    endpoints = pair_endpoint_targets(tables['pairs','train']) if args.pair_absolute_weight else None
    if not all(torch.isfinite(t).all() for ts in tensors.values() for t in ts):
        raise ValueError('Nonfinite data')
    # Both modes fit the same statistics, using absolute train inputs only.
    unique_train = sorted({r['feature_path'] for r in tables['absolute','train']})
    train_features = torch.stack([features[path] for path in unique_train])
    y = tensors['absolute','train'][-1]
    model = CachedAffinityRegressor(train_features.mean(0),train_features.std(0,unbiased=False).clamp_min(1e-6),
                                    y.mean(),y.std(unbiased=False).clamp_min(1e-6)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ['resume']}
    config.update(input_hashes={f.name:digest(f) for f in sorted(args.data.glob('*.csv'))},
                  feature_hash=feature_hash,model_hash=digest(ROOT/'appi/model.py'),
                  trainer_hash=digest(Path(__file__)),torch_version=str(torch.__version__),
                  selection='minimum absolute validation RMSE; test only after selection',
                  normalization='unique absolute training features; absolute training targets',
                  objective='L_abs + pair_absolute_weight * mean_pair_endpoint_MSE '
                            '+ (lambda_diff if joint else 0) * L_diff; all scaled by target variance',
                  pair_endpoint_sampling='uniform pair experiments, WT/mutant each half weight; '
                                         'repeated endpoints retain their experimental labels')
    config_path=args.out/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text())!=config: raise ValueError('Config changed: use a new output directory')
        if not args.resume: raise ValueError('Output exists; use --resume or a new directory')
    else: json_save(config,config_path)
    start,best,stale = 0,float('inf'),0
    if args.resume:
        ckpt=torch.load(args.out/'last.pt',map_location=device,weights_only=True)
        model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer'])
        start,best,stale=ckpt['epoch'],ckpt['best'],ckpt['stale']
        torch.set_rng_state(ckpt['rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    a=tensors['absolute','train']; d=tensors['pairs','train']
    for epoch in range(start,args.epochs):
        if stale>=args.patience: break
        model.train(); permutation=torch.randperm(len(a[-1]));losses=[]
        for offset in range(0,len(permutation),args.batch_size):
            ids=permutation[offset:offset+args.batch_size]
            x, target=(t[ids].to(device) for t in a)
            abs_loss=((model(x)-target)/model.target_scale).square().mean()
            diff_loss=abs_loss.new_zeros(())
            pair_abs_loss=abs_loss.new_zeros(())
            if args.mode=='joint' or args.pair_absolute_weight:
                # Uniform draws give the mean pair loss equal expected weight each step.
                # With endpoint supervision both modes consume identical RNG draws.
                dids=torch.randint(len(d[-1]),(len(ids),))
                mut,wt,delta=(t[dids].to(device) for t in d)
                mut_pred,wt_pred=model(mut),model(wt)
                if endpoints is not None:
                    pair_abs_loss,pair_diff_loss=pair_losses(
                        mut_pred,wt_pred,endpoints[dids].to(device),delta,model.target_scale)
                else:
                    pair_diff_loss=((mut_pred-wt_pred-delta)/model.target_scale).square().mean()
                if args.mode=='joint': diff_loss=pair_diff_loss
            loss=abs_loss+args.pair_absolute_weight*pair_abs_loss+args.lambda_diff*diff_loss
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            losses.append([float(abs_loss.detach()),float(diff_loss.detach()),len(ids),
                           float(pair_abs_loss.detach())])
        av,_=evaluate(model,tensors['absolute','val'],device,args.batch_size)
        dv,_=evaluate(model,tensors['pairs','val'],device,args.batch_size)
        improved=av['rmse']<best
        if improved: best,stale=av['rmse'],0
        else: stale+=1
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
               'best':best,'stale':stale,'rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'config':config}
        if improved:atomic_save(state,args.out/'best.pt')
        atomic_save(state,args.out/'last.pt')
        record={'epoch':epoch+1,'train_abs_mse_scaled':sum(x[0]*x[2] for x in losses)/len(y),
                'train_diff_mse_scaled':sum(x[1]*x[2] for x in losses)/len(y),
                'train_pair_abs_mse_scaled':sum(x[3]*x[2] for x in losses)/len(y),
                'val_absolute':av,'val_delta_pkd':dv,'best_epoch_updated':improved}
        # One epoch per file makes a resumed run replace any uncommitted log entry.
        json_save(record,args.out/f'epoch_{epoch+1:04d}.json')
        print(json.dumps(record),flush=True)
    best_ckpt=torch.load(args.out/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(best_ckpt['model'])
    result={'mode':args.mode,'seed':args.seed,'best_epoch':best_ckpt['epoch'],
            'smoke_test':bool(args.limit),'metrics':{}}
    for split in ['val','test']:
        for kind in ['absolute','pairs']:
            m,pred=evaluate(model,tensors[kind,split],device,args.batch_size)
            result['metrics'][kind+'_'+split]=m
            with (args.out/f'predictions_{kind}_{split}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['source_line','split','target','prediction','unit'])
                w.writeheader()
                for r,t,v in zip(tables[kind,split],tensors[kind,split][-1].tolist(),pred):
                    w.writerow(dict(source_line=r['csv_line' if kind=='absolute' else 'skempi_csv_line'],
                                    split=split,target=t,prediction=v,unit='pKd' if kind=='absolute' else 'delta_pKd'))
    json_save(result,args.out/'metrics.json')
    print('Completed: '+str(args.out/'metrics.json'),flush=True)


if __name__=='__main__':main()
