"""Export the explicitly chosen seed44 research candidate, without retraining."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.structure_context_data import file_digest
from appi.structure_context_model import StructureContextRegressor

VERSION = 'v0.1.0-rc1-seed44'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/releases'/VERSION)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new export directory; existing releases are not overwritten')
    run = ROOT/'runs/structure_context_pilot_v1/fusion_transfer_seed44'
    checkpoint = run/'joint/best.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = json.loads((run/'config.json').read_text())
    assert saved['config'] == config and saved['epoch'] == 16 and saved['stage'] == 'joint'
    for path, expected in config['code_hashes'].items():
        assert file_digest(ROOT/path) == expected, path
    model = StructureContextRegressor(**config['model'])
    model.load_state_dict(saved['model'])
    ppi = json.loads((ROOT/'data/processed/adaptyv_egfr_features_v1/config.json').read_text())
    exported = dict(version=VERSION, seed=44, selected_epoch=16, model=config['model'],
        parameters=config['parameters'], esm_extraction=config['esm']['extraction'],
        ppi_feature_requirements={k: ppi[k] for k in ('ppi_checkpoint_sha256', 'radius_angstrom', 'interface_code_sha256', 'ppi_device')},
        checkpoint_selection=config['selection'],
        seed_selection='User selected seed44 after reviewing internal and EGFR results; not a prospectively selected seed.',
        output='predicted pKd = -log10(Kd/M); not a calibrated binding probability',
        source_checkpoint_sha256=file_digest(checkpoint), training_config_sha256=file_digest(run/'config.json'),
        source_code_sha256=config['code_hashes'])
    args.out.mkdir(parents=True)
    (args.out/'appi').mkdir()
    (args.out/'scripts').mkdir()
    for path in ('appi/structure_context_model.py', 'appi/structure_context_data.py', 'scripts/predict_release.py'):
        shutil.copy2(ROOT/path, args.out/path)
    (args.out/'appi/__init__.py').write_text('')
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, args.out/'model.pt')
    # Buffers retain the original training normalization; no external-set recalibration.
    (args.out/'config.json').write_text(json.dumps(exported, indent=2)+'\n')
    shutil.copy2(ROOT/'releases'/VERSION/'MODEL_CARD.md', args.out/'MODEL_CARD.md')
    shutil.copy2(ROOT/'releases'/VERSION/'INFERENCE.md', args.out/'INFERENCE.md')
    (args.out/'requirements.txt').write_text('torch=='+torch.__version__.split('+')[0]+'\n')
    hashes = {str(p.relative_to(args.out)): file_digest(p) for p in sorted(args.out.rglob('*')) if p.is_file()}
    manifest = dict(version=VERSION, files_sha256=hashes,
                    source_checkpoint_sha256=file_digest(checkpoint), third_party_encoder_weights_included=False)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = args.out.with_name(args.out.name + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as f:
        f.add(args.out, arcname=VERSION)
    checksum = file_digest(archive)
    archive.with_name(archive.name+'.sha256').write_text(checksum+'  '+archive.name+'\n')
    print(json.dumps(dict(directory=str(args.out), archive=str(archive), sha256=checksum), indent=2))


if __name__ == '__main__':
    main()
