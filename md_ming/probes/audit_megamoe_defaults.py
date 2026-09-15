#!/usr/bin/env python3
"""Check the actual EP8 selector against the retained 2026-09-15 ledger; CPU only."""
from pathlib import Path
import ast,dataclasses,importlib.util,json,os,sys
from types import SimpleNamespace
from unittest.mock import patch
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path/'mega_moe_config.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 cls=next(n for n in ast.parse((path/'mega_moe_m3.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MegaMoEM3')
 fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_select_config');ns=dict(vars(m));exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
 guards={}
 for field in ('_sbm32_scope','_sbm64_scope','_sbm128_scope'):
  expr=next(n.value for n in ast.walk(cls) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr==field for t in n.targets));guards[field]=compile(ast.Expression(expr),field,'eval')
 def choose(mtpr,tokens,**changed):
  attrs=dict(mtpr=mtpr,world_size=8,epr=16,model_dim=6144,inter_dim=3072,topk=4,local_reduce=False,local_reduce_xcd_local=False,_s1_fixed_slot=False,_shared_l13=1);attrs.update(changed);op=SimpleNamespace(**attrs)
  env=dict(self=op,shared_w13=op._shared_l13,shared_w2=changed.get('shared_w2',1),shared_xcd_schedule=changed.get('shared_xcd_schedule',True))
  for f,g in guards.items():setattr(op,f,eval(g,vars(m),env))
  return dataclasses.asdict(ns['_select_config'](op,tokens))
 return m,choose

if __name__ == '__main__':
 import argparse
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',type=Path,required=True);args=ap.parse_args()
 pkg=args.repo.resolve()/'aiter/ops/flydsl/kernels/mega_moe_m3'
 cfg,choose=load(pkg,'megamoe_default_audit_config')
 names=list(cfg._STAGE1_OVERRIDE_ENV.values())+[cfg._P2P_QUANT_ENV]
 active=[n for n in names if os.environ.get(n)]
 if active:raise SystemExit('Clear explicit overrides before checking defaults: '+', '.join(active))
 ledger=json.loads(Path(__file__).with_name('megamoe_defaults.json').read_text())
 for t,expected in ledger['configs'].items():
  actual=choose(int(t),int(t))
  if actual!=expected:raise SystemExit('Default config mismatch: batch '+t)
 print('PASS: '+str(len(ledger['configs']))+' EP8 full-batch defaults match the retained ledger')
