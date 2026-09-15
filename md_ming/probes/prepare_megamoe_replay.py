#!/usr/bin/env python3
"""Prepare a new frozen replay; never alter the historical experiment or its pins."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--experiment', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
src, dst = args.experiment.resolve(), args.out.resolve()
allowed_names = {f'b{t}_s2_m64_backward_20260915_v{2 if t == 184 else 1}' for t in range(120,185,8)} | {'b112_s2_epi16_20260915_v3'}
if src.name not in allowed_names or dst.exists() or src in dst.parents:
    raise SystemExit('Use a supported historical experiment and a fresh output directory outside it')
rebase = json.loads(Path(__file__).with_name('megamoe_replay_rebase.json').read_text())
pins = json.loads((src/'measure/source_manifest.json').read_text())['sources']
changed = {}
# Fail closed before copying. Only this rollout's independently audited source pairs may differ.
for name, expected in pins.items():
    p = Path(name)
    actual = sha(p)
    if actual != expected:
        if name not in rebase or {'before': expected, 'after': actual} not in rebase[name]:
            raise SystemExit('Unreviewed source drift: '+name)
        changed[name] = dict(before=expected, after=actual)
files = {Path(name) for name in pins if Path(name).is_relative_to(src)}
files.update(p for p in (src/'review').glob('*.py'))
files.add(src/'measure/run.sh')
dst.mkdir(parents=True)
for p in files:
    q = dst/p.relative_to(src)
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p,q)
# Relocate paths only when the corresponding local file was copied. External references remain pinned.
def relocate(text):
    return text.replace(str(src),str(dst))
for p in files:
    q=dst/p.relative_to(src)
    if q.suffix in ('.py','.json','.sh','.diff'):
        q.write_text(relocate(q.read_text()))
        if q.suffix=='.py': ast.parse(q.read_text())
external_path=dst/'review/external_sources.json'
external=json.loads(external_path.read_text())
external_path.write_text(json.dumps({name:sha(Path(name)) for name in external},indent=2)+'\n')
# The old harness carried a stale literal commit label. Correct metadata for this replay.
repo = Path(next(iter(rebase))).parents[5]
head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
bench = dst/'measure/bench.py'
s = bench.read_text()
s = re.sub(r"(manifest\['current_production_commit'\] = )'[^']*'",lambda m:m[1]+repr(head),s)
bench.write_text(s)
expected_path=dst/'review/expected_run.json'
expected=json.loads(expected_path.read_text())
expected['harness']=str(bench);expected['harness_sha256']=sha(bench)
expected_path.write_text(json.dumps(expected,indent=2)+'\n')
# Update the declared snapshot/current diff list, without altering the measured snapshots.
variants_path=dst/'measure/variants.json'
variants=json.loads(variants_path.read_text())
production_kernels=Path(next(iter(rebase))).parent.parent
for name, entry in variants.items():
    folder=dst/'measure'/name
    if folder.is_dir():
        entry['differs']=sorted(str(p.relative_to(folder)) for p in folder.rglob('*.py')
                                if sha(p)!=sha(production_kernels/p.relative_to(folder)))
variants_path.write_text(json.dumps(variants,indent=2)+'\n')
# Preserve the historical spec's provenance and record the current replay separately.
record=dict(original_experiment=str(src),original_manifest_sha256=sha(src/'measure/source_manifest.json'),
            replay_production_commit=head,reviewed_external_rebase=changed,
            reason='Identical full default configurations and S2 ISA verified in s2_defaults_20260915; only symbol names differ for M64 scans.',
            gpu_run=False)
(dst/'review/replay_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
newpins={}
for name,expected_hash in pins.items():
    p=Path(name)
    q=dst/p.relative_to(src) if p.is_relative_to(src) else p
    newpins[str(q)]=sha(q)
for p in files:
    q=dst/p.relative_to(src);newpins[str(q)]=sha(q)
newpins[str(dst/'review/replay_provenance.json')]=sha(dst/'review/replay_provenance.json')
manifest=dst/'measure/source_manifest.json'
manifest.write_text(json.dumps(dict(sources=newpins),indent=2)+'\n')
assert all(sha(Path(p))==h for p,h in newpins.items())
# Do not invoke freeze.py: it would replace the reviewed replay ledger with an old preregistration.
print('PREPARED',dst,'PINS',len(newpins),'REBASED',len(changed),'GPU_NOT_RUN')
