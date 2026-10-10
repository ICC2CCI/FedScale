# FineWeb-Edu 联邦续预训练切分（STAGE-PT）

- **dataset**: `HuggingFaceFW/fineweb-edu`
- **revision**: `main`
- **source**: `{'mode': 'parquet', 'files': ['data/cache/fineweb-edu-10bt/000_00000.parquet']}`
- **seed**: `20260831`
- **token_count**: `chars/4`
- **train tokens（合计）**: 500,002,073
- **eval tokens**: 50,000,521
- **icc1 / icc2 docs**: 210,401 / 210,402
- **切法**: 文档级随机 50/50（IID proxy）

| 文件 | 用途 |
|---|---|
| `icc1_train.jsonl` | ICC1 / client0 |
| `icc2_train.jsonl` | ICC2 / client1 |
| `eval.jsonl` | 共享 hold-out（PPL/CE） |
| `manifest.json` | 元信息 |

详见 `docs/exec-plans/active/2026-10-08-stage-pt-federated-continued-pretrain.md`。
