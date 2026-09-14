"""CPU-only regression tests for the measured EP8 SBM64/SBM128 default boundaries."""
import ast
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


KERNELS = Path(os.environ.get('M3_KERNEL_ROOT',
    Path(__file__).resolve().parents[1] / 'aiter/ops/flydsl/kernels'))
spec = importlib.util.spec_from_file_location('m3_cpu_config', KERNELS / 'mega_moe_m3/mega_moe_config.py')
cfg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cfg
spec.loader.exec_module(cfg)
tree = ast.parse((KERNELS / 'mega_moe_m3/mega_moe_m3.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MegaMoEM3')
selector = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_select_config')
namespace = dict(vars(cfg))
exec(compile(ast.Module(body=[selector], type_ignores=[]), '<actual selector>', 'exec'), namespace)
scope_expr = next(n.value for n in ast.walk(cls) if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Attribute) and t.attr == '_sbm128_scope' for t in n.targets))
scope_code = compile(ast.Expression(scope_expr), '<actual scope guard>', 'eval')
scope64_expr = next(n.value for n in ast.walk(cls) if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Attribute) and t.attr == '_sbm64_scope' for t in n.targets))
scope64_code = compile(ast.Expression(scope64_expr), '<actual SBM64 scope guard>', 'eval')


def operator(tokens, *, shared=True, shared_l2=True, xcd=True, world=8, epr=16, hidden=6144, inter=3072, topk=4, lr=False):
    op = SimpleNamespace(mtpr=tokens, world_size=world, epr=epr, model_dim=hidden,
        inter_dim=inter, topk=topk, local_reduce=lr, local_reduce_xcd_local=False,
        _s1_fixed_slot=False, _shared_l13=1 if shared else None)
    op._sbm128_scope = eval(scope_code, vars(cfg), dict(self=op,
        shared_w13=1 if shared else None, shared_w2=1 if shared_l2 else None, shared_xcd_schedule=xcd))
    op._sbm64_scope = eval(scope64_code, vars(cfg), dict(self=op,
        shared_w13=1 if shared else None, shared_w2=1 if shared_l2 else None, shared_xcd_schedule=xcd))
    return op


class SBM128DefaultsTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {n: '' for n in cfg._STAGE1_OVERRIDE_ENV.values()})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_all_measured_defaults(self):
        self.assertEqual(set(cfg.SBM128_PATHS), set(range(200, 257, 8)))
        for tokens in cfg.SBM128_PATHS:
            with self.subTest(tokens=tokens):
                op = operator(tokens)
                value = namespace['_select_config'](op, tokens)
                self.assertTrue(op._sbm128_scope)
                self.assertEqual(value.stage1.sbm128_path, 'tiered96' if tokens >= 224 else 'm16')
                self.assertEqual((value.stage1.sort_block_m, value.stage1.tile_n, value.stage1.tile_k), (128, 256, 256))
                self.assertEqual(value.stage1.num_dispatch_cu, 32 if tokens == 256 else 48)
                self.assertEqual(value.stage1.b_nt, 0 if tokens == 256 else 2)
                self.assertEqual(value.stage1.preplan_waves, 4)
                self.assertEqual((value.stage2.block_m, value.stage2.block_n), (32, 128 if tokens == 256 else 256))
                self.assertEqual(value.stage2.shared_schedule, 'jointtail')
                self.assertTrue(cfg.shared_l2_s2_shape_ok(tokens, 32, value.stage2.block_n, 256, 128))

    def test_scope_excludes_unmeasured_workloads(self):
        for tokens in cfg.SBM128_PATHS:
            for changed in (dict(shared=False), dict(shared_l2=False), dict(xcd=False), dict(world=4),
                            dict(epr=32), dict(hidden=4096), dict(inter=4096), dict(topk=2), dict(lr=True)):
                with self.subTest(tokens=tokens, changed=changed):
                    self.assertFalse(operator(tokens, **changed)._sbm128_scope)
        for tokens in (192, 199, 201, 207, 209, 255, 257, 512, 8192):
            self.assertFalse(operator(tokens)._sbm128_scope)

    def test_generic_small_sort_table_unchanged(self):
        for tokens in range(8, 257, 8):
            expected = 64 if tokens % 64 == 0 or tokens > 96 else 32
            self.assertEqual(cfg.SHARED_SMALL_SORT_BLOCK_M[tokens], expected)
        generic = namespace['_select_config'](operator(192), 192).stage1
        self.assertEqual(generic.sbm128_path, 'generic')
        with self.assertRaises(ValueError):
            replace(generic, sbm128_path='m16', sort_block_m=64, tile_n=256)

    def test_partial_batch_does_not_select_specialized_path(self):
        for mtpr in cfg.SBM128_PATHS:
            value = namespace['_select_config'](operator(mtpr), 192)
            self.assertEqual(value.stage1.sbm128_path, 'generic')

    def test_explicit_environment_tuning_bypasses_rollout(self):
        for name in cfg._STAGE1_OVERRIDE_ENV.values():
            with patch.dict(os.environ, {name: '1'}):
                self.assertFalse(operator(200)._sbm128_scope)


class SBM64DefaultsTest(unittest.TestCase):
    setUp = SBM128DefaultsTest.setUp

    def test_corrected_baseline_defaults(self):
        paths = {120: 'full', 136: 'full', 160: 'm16_dma', 168: 'm16_dma',
                 176: 'm16_dma', 184: 'm16_dma', 192: 'm16'}
        self.assertEqual(cfg.SBM64_PATHS, paths)
        for tokens in range(8, 257, 8):
            value = namespace['_select_config'](operator(tokens), tokens)
            self.assertEqual(value.stage1.sbm64_path, paths.get(tokens, 'generic'))
            if tokens in paths:
                self.assertEqual((value.stage1.sort_block_m, value.stage1.tile_n), (64, 512))
                self.assertEqual((value.stage2.block_m, value.stage2.block_n), (32, 256))
                self.assertEqual(value.stage1.sbm128_path, 'generic')

    def test_sbm64_scope_boundaries(self):
        for tokens in cfg.SBM64_PATHS:
            for changed in (dict(shared=False), dict(shared_l2=False), dict(xcd=False), dict(world=4),
                            dict(epr=32), dict(hidden=4096), dict(inter=4096), dict(topk=2), dict(lr=True)):
                self.assertFalse(operator(tokens, **changed)._sbm64_scope)
            self.assertEqual(namespace['_select_config'](operator(tokens), 64).stage1.sbm64_path, 'generic')
            for name in cfg._STAGE1_OVERRIDE_ENV.values():
                with patch.dict(os.environ, {name: '1'}):
                    self.assertFalse(operator(tokens)._sbm64_scope)
        value = namespace['_select_config'](operator(160), 160).stage1
        for changes in (dict(tile_n=256), dict(sort_block_m=128), dict(sbm128_path='m16'), dict(sbm64_path='bad')):
            with self.assertRaises(ValueError):
                replace(value, **changes)


if __name__ == '__main__':
    unittest.main()
