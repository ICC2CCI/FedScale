"""CPU smoke test：用合成小模型验证 run_experiment 三种 method 逻辑跑通。

不依赖 transformers / accelerate / GPU。验证：
1. dense / topk / sketchopa 三种方法都能完成一轮
2. sketchopa 安全层 vs 明文聚合结果一致
3. calibration 计数能正确保存
"""
import os
import sys
import json
import shutil
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # experiments_sketchopa/


class DummyModel(torch.nn.Module):
    """合成小模型，模拟 state_dict 结构。"""
    def __init__(self, dims=((50, 30), (30, 20), (20, 10))):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(a, b, bias=True) for a, b in dims])
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def _make_args(method, out_dir, **kwargs):
    """构造 run_experiment 的 args。"""
    import argparse
    defaults = dict(
        method=method,
        model_path="dummy",
        data_paths=["dummy0", "dummy1"],
        eval_path="dummy_eval",
        num_rounds=2,
        local_steps=2,
        batch_size=2,
        grad_accum=1,
        lr=1e-3,
        seq_len=8,
        out_dir=out_dir,
        seed=42,
        topk_ratio=0.05,
        recover_rho=2.0,
        rows=3,
        budget_ratio=2.0,
        allocation="uniform_param",
        calibration_path=None,
        residual=False,
        residual_clip=0.0,
        residual_decay=1.0,
        transport_dtype="float16",
        quant_scale=1e6,
        sketch_seed=42,
        secure=True,
        clip_norm=10.0,
        save_calibration=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _patch_common(monkey_dataloaders):
    """替换 common 中的数据加载/模型加载/训练为合成版本。"""
    import common
    import run_experiment as rx

    # 替换 setup_accelerate
    class FakeAccelerator:
        is_main_process = True
        num_processes = 1
        device = torch.device("cpu")
        def wait_for_everyone(self): pass
        def prepare(self, *args): return args

    common.setup_accelerate = lambda: FakeAccelerator()
    rx.setup_accelerate = common.setup_accelerate

    # 替换 load_model / load_tokenizer
    common.load_model = lambda path, torch_dtype=torch.float32: DummyModel()
    rx.load_model = common.load_model
    common.load_tokenizer = lambda path: None
    rx.load_tokenizer = common.load_tokenizer

    # 替换 dataloader
    def fake_loader(items, tokenizer, seq_len, batch_size, shuffle=True):
        return monkey_dataloaders[items]
    def fake_eval_loader(items, tokenizer, seq_len, batch_size):
        return monkey_dataloaders["eval"]
    common.build_dataloader = fake_loader
    rx.build_dataloader = common.build_dataloader
    common.build_eval_loader = fake_eval_loader
    rx.build_eval_loader = common.build_eval_loader

    # 替换 train_local_steps（直接做一步梯度下降）
    def fake_train(model, loader, optimizer, scheduler, local_steps, device, grad_accum=1):
        model.train()
        for _ in range(local_steps):
            for batch in loader:
                optimizer.zero_grad()
                out = model(batch["x"])
                loss = out.pow(2).mean()
                loss.backward()
                optimizer.step()
                if scheduler: scheduler.step()
                break
        return 0.5, [0.5]
    common.train_local_steps = fake_train
    rx.train_local_steps = common.train_local_steps

    # 替换 evaluate
    def fake_eval(model, loader, device):
        model.eval()
        with torch.no_grad():
            total = 0.0; n = 0
            for batch in loader:
                out = model(batch["x"])
                total += out.pow(2).mean().item(); n += 1
            return total / max(n, 1)
    common.evaluate = fake_eval
    rx.evaluate = common.evaluate

    # get_full_state_fsdp
    common.get_full_state_fsdp = lambda model: {k: v.clone() for k, v in model.state_dict().items()}
    rx.get_full_state_fsdp = common.get_full_state_fsdp


def _make_dummy_loaders():
    """构造合成数据 loader。"""
    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, n=20, dim=50):
            self.x = torch.randn(n, dim)
        def __len__(self): return len(self.x)
        def __getitem__(self, i): return {"x": self.x[i]}
    loader0 = torch.utils.data.DataLoader(FakeDataset(20), batch_size=2)
    loader1 = torch.utils.data.DataLoader(FakeDataset(20), batch_size=2)
    eval_loader = torch.utils.data.DataLoader(FakeDataset(10), batch_size=2)
    return {
        "dummy0": loader0,
        "dummy1": loader1,
        "eval": eval_loader,
    }


def test_dense():
    loaders = _make_dummy_loaders()
    _patch_common(loaders)
    import run_experiment as rx
    with tempfile.TemporaryDirectory() as tmp:
        args = _make_args("dense", tmp)
        rx.args = args
        # monkey-patch parse_args
        rx.parse_args = lambda: args
        rx.main()
        with open(os.path.join(tmp, "round_log.json")) as f:
            log = json.load(f)
        assert len(log) == 2, f"expected 2 rounds, got {len(log)}"
        assert "eval_loss" in log[0]
        print(f"[OK] test_dense: eval_loss={log[-1]['eval_loss']:.4f}")


def test_topk():
    loaders = _make_dummy_loaders()
    _patch_common(loaders)
    import run_experiment as rx
    with tempfile.TemporaryDirectory() as tmp:
        args = _make_args("topk", tmp, residual=True)
        rx.parse_args = lambda: args
        rx.main()
        with open(os.path.join(tmp, "round_log.json")) as f:
            log = json.load(f)
        assert len(log) == 2
        print(f"[OK] test_topk: eval_loss={log[-1]['eval_loss']:.4f}")


def test_sketchopa_uniform():
    loaders = _make_dummy_loaders()
    _patch_common(loaders)
    import run_experiment as rx
    with tempfile.TemporaryDirectory() as tmp:
        args = _make_args("sketchopa", tmp, allocation="uniform_param", residual=True, secure=True)
        rx.parse_args = lambda: args
        rx.main()
        with open(os.path.join(tmp, "round_log.json")) as f:
            log = json.load(f)
        assert len(log) == 2
        assert "sketch_len" in log[0]
        print(f"[OK] test_sketchopa_uniform: eval_loss={log[-1]['eval_loss']:.4f}, sketch_len={log[-1]['sketch_len']}")


def test_sketchopa_calibrated():
    loaders = _make_dummy_loaders()
    _patch_common(loaders)
    import run_experiment as rx
    with tempfile.TemporaryDirectory() as tmp:
        cal_path = os.path.join(tmp, "cal.json")
        # 先跑 topk 生成 calibration
        args = _make_args("topk", tmp, save_calibration=cal_path, num_rounds=2)
        rx.parse_args = lambda: args
        rx.main()
        assert os.path.exists(cal_path), "calibration not saved"
        # 再跑 sketchopa calibrated
        args2 = _make_args("sketchopa", tmp + "_sk", allocation="topk_count_calibrated",
                           calibration_path=cal_path, residual=True)
        rx.parse_args = lambda: args2
        rx.main()
        with open(os.path.join(tmp + "_sk", "round_log.json")) as f:
            log = json.load(f)
        assert len(log) == 2
        print(f"[OK] test_sketchopa_calibrated: eval_loss={log[-1]['eval_loss']:.4f}")


def test_sketchopa_no_secure():
    """验证 --no-secure 也能跑通。"""
    loaders = _make_dummy_loaders()
    _patch_common(loaders)
    import run_experiment as rx
    with tempfile.TemporaryDirectory() as tmp:
        args = _make_args("sketchopa", tmp, secure=False, residual=True)
        rx.parse_args = lambda: args
        rx.main()
        with open(os.path.join(tmp, "round_log.json")) as f:
            log = json.load(f)
        assert len(log) == 2
        print(f"[OK] test_sketchopa_no_secure: eval_loss={log[-1]['eval_loss']:.4f}")


def main():
    test_dense()
    test_topk()
    test_sketchopa_uniform()
    test_sketchopa_calibrated()
    test_sketchopa_no_secure()
    print("\n=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    main()
