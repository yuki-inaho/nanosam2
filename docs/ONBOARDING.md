# 環境セットアップ・オンボーディングガイド

**作成日**: `2026-06-30`
**対象**: `新しいセッション・エージェント、開発メンバー`
**プロジェクト**: `nanosam2`
**目的**: `SAM2.1 / nanosam2 の ONNX export 作業環境を uv で再現し、image encoder / mask decoder の export と検証を継続できるようにする。`

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [現在のプロジェクト状態](#2-現在のプロジェクト状態)
3. [前提条件の確認](#3-前提条件の確認)
4. [環境セットアップ手順](#4-環境セットアップ手順)
5. [動作確認](#5-動作確認)
6. [トラブルシューティング](#6-トラブルシューティング)
7. [次のステップ](#7-次のステップ)
8. [環境セットアップ完了チェックリスト](#8-環境セットアップ完了チェックリスト)
9. [更新履歴](#9-更新履歴)

---

## 1. プロジェクト概要

### プロジェクト名

**nanosam2**
`SAM2.1 系モデルを軽量バックボーンや ONNX export に対応させるための実験・変換用リポジトリ。`

### 最終目標

SAM2.1 / nanosam2 の image encoder と mask decoder を、再現可能な uv 環境で ONNX へ export する。
生成した ONNX を `onnx.checker` と `onnxruntime` で検証し、後段のバッチセグメンテーション検証に利用できる状態にする。

### 主要コンポーネント

* `tools/onnx_export.py` - SAM2.1 / nanosam2 モデルブロックの ONNX export CLI
* `sam2_configs/` - SAM2.1 / nanosam2 の Hydra config
* `nanosam2/sam2/` - SAM2.1 モデル実装
* `pyproject.toml` / `uv.lock` - uv ベースの Python 環境定義
* `model_exports2/` - ONNX 出力先。git 管理外
* `results/` - checkpoint 配置先。git 管理外

---

## 2. 現在のプロジェクト状態

### 完了済み

| 分類 | 状態 | 説明 |
| --- | --- | --- |
| **PR #6 取り込み** | 完了 | ONNX export 用の変更をローカル main に取り込み済み。 |
| **uv 化** | 完了 | `[project]` と依存関係を `pyproject.toml` に追加し、`uv.lock` を生成済み。 |
| **Python バージョン固定** | 完了 | `.python-version` で Python 3.12 を指定。 |
| **SAM2.1 Hiera small checkpoint** | 完了 | `results/sam2.1_hiera_s/sam2.1_hiera_small.pt` に配置。git 管理外。 |
| **image encoder export** | 完了 | `sam2.1_small-image-encoder-sa1-v01-op17.onnx` を生成・検証済み。 |
| **mask decoder export** | 完了 | `sam2.1_small-mask-decoder-sa1-v01-op17.onnx` を生成・検証済み。 |
| **ONNX simplifier** | 完了 | `onnxsim` を依存関係に追加し、`-simplified.onnx` 生成を確認済み。 |

### 依存パッケージのインストール状態

新しい checkout では `.venv` が存在しない可能性がある。必ず `uv sync` で `uv.lock` に基づく環境を再現する。
依存を追加する場合は `uv add <package>` を使う。`uv pip install` は使わない。

### 未実装・これから着手する項目

* 実画像・プロンプト点・BBOX を使った end-to-end ONNX 推論パイプライン
* image encoder の出力名整理と後段 decoder 入力への mapping 明文化
* GPU Execution Provider での検証。古い GPU では CUDA wheel 非対応の可能性がある
* nanosam2-resnet18 / mobilenetV3 checkpoint を用いた軽量 encoder export

### 重要なファイル／ディレクトリ

```text
<repo-root>/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── docs/
│   └── ONBOARDING.md
├── tools/
│   └── onnx_export.py
├── sam2_configs/
│   ├── sam2.1_hiera_s.yaml
│   ├── nanosam2.1_resnet18.yaml
│   └── nanosam2.1_mobilenet_v3_large.yaml
├── nanosam2/
│   └── sam2/
├── results/
│   └── sam2.1_hiera_s/sam2.1_hiera_small.pt      # git 管理外
└── model_exports2/
    ├── sam2.1_small-image-encoder-sa1-v01-op17.onnx              # git 管理外
    ├── sam2.1_small-image-encoder-sa1-v01-op17-simplified.onnx   # git 管理外
    ├── sam2.1_small-mask-decoder-sa1-v01-op17.onnx               # git 管理外
    └── sam2.1_small-mask-decoder-sa1-v01-op17-simplified.onnx    # git 管理外
```

---

## 3. 前提条件の確認

新しいセッションで環境を再現する前に、以下を確認する。

### 3.1 システム情報の確認

```bash
cat /etc/os-release | grep -E "^(NAME|VERSION)="
uname -r
nproc
free -h
pwd
```

### 3.2 必須ツールの存在確認

```bash
which uv && uv --version
which git && git --version
which wget && wget --version | head -1
```

`uv` がない場合は、公式手順に従ってインストールする。

### 3.3 Git ブランチ・コミット確認

```bash
git branch --show-current
git log --oneline -1
git status --short --branch
```

---

## 4. 環境セットアップ手順

### 4.1 Python 環境の再現

```bash
cd <repo-root>
uv python pin 3.12
SAM2_BUILD_CUDA=0 uv sync
```

`SAM2_BUILD_CUDA=0` は、SAM2 の CUDA extension build を避けるために指定する。ONNX export 自体は CPU で実行できる。

### 4.2 依存パッケージの追加方法

依存追加は必ず `uv add` を使う。

```bash
SAM2_BUILD_CUDA=0 uv add onnxsim
SAM2_BUILD_CUDA=0 uv add onnx onnxruntime opencv-python
```

既存環境の import 確認は以下で行う。

```bash
SAM2_BUILD_CUDA=0 uv run python - <<'PY'
import torch
import onnx
import onnxruntime as ort
import onnxsim
print('torch', torch.__version__)
print('onnx', onnx.__version__)
print('onnxruntime', ort.__version__, ort.get_available_providers())
print('onnxsim', getattr(onnxsim, '__version__', 'unknown'))
PY
```

### 4.3 Checkpoint の取得

SAM2.1 Hiera small の checkpoint は git 管理外の `results/` に置く。

```bash
mkdir -p results/sam2.1_hiera_s
wget -O results/sam2.1_hiera_s/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

### 4.4 ONNX export

image encoder:

```bash
SAM2_BUILD_CUDA=0 uv run python tools/onnx_export.py \
  --export image-encoder \
  --encoder_type hiera_small \
  --img_shape 3 1024 1024 \
  --opset 17 \
  --output_path model_exports2
```

mask decoder:

```bash
SAM2_BUILD_CUDA=0 uv run python tools/onnx_export.py \
  --export mask-decoder \
  --encoder_type hiera_small \
  --img_shape 3 1024 1024 \
  --opset 17 \
  --output_path model_exports2
```

ONNX simplifier を使う場合:

```bash
SAM2_BUILD_CUDA=0 uv run python tools/onnx_export.py \
  --export mask-decoder \
  --encoder_type hiera_small \
  --img_shape 3 1024 1024 \
  --opset 17 \
  --output_path model_exports2 \
  --simplify \
  --simplify-tensor-size-threshold 1MB
```

注意: image encoder は simplification によりファイルサイズが大きくなる場合がある。速度も必ずしも改善しないため、実運用では original と simplified の両方をベンチして選ぶ。

---

## 5. 動作確認

### 5.1 ONNX checker と ORT load

```bash
SAM2_BUILD_CUDA=0 uv run python - <<'PY'
from pathlib import Path
import onnx
import onnxruntime as ort

for p in sorted(Path('model_exports2').glob('sam2.1_small-*-op17*.onnx')):
    model = onnx.load(p)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(p), providers=['CPUExecutionProvider'])
    print(p.name)
    print('  inputs ', [(i.name, i.shape, i.type) for i in session.get_inputs()])
    print('  outputs', [(o.name, o.shape, o.type) for o in session.get_outputs()])
PY
```

### 5.2 ダミー推論

```bash
SAM2_BUILD_CUDA=0 uv run python - <<'PY'
from pathlib import Path
import numpy as np
import onnxruntime as ort

for p in sorted(Path('model_exports2').glob('sam2.1_small-*-op17*.onnx')):
    session = ort.InferenceSession(str(p), providers=['CPUExecutionProvider'])
    feed = {i.name: np.zeros([int(v) for v in i.shape], dtype=np.float32) for i in session.get_inputs()}
    outputs = session.run(None, feed)
    print(p.name, [o.shape for o in outputs])
PY
```

### 5.3 期待される代表出力

image encoder:

```text
input:  [1, 3, 1024, 1024]
output: [1, 256, 256, 256], [1, 256, 128, 128], [1, 256, 64, 64] ...
```

mask decoder:

```text
inputs:
  image_embeddings         [1, 256, 64, 64]
  image_pe                 [1, 256, 64, 64]
  sparse_prompt_embeddings [1, 8, 256]
  dense_prompt_embeddings  [1, 256, 64, 64]
  high_res_feature_0       [1, 32, 256, 256]
  high_res_feature_1       [1, 64, 128, 128]
outputs:
  mask logits              [1, 1, 256, 256]
  iou prediction           [1, 1]
  mask token               [1, 1, 256]
  object score             [1, 1]
```

---

## 6. トラブルシューティング

### 問題1: `uv` が見つからない

**原因**: uv が未インストール、または PATH が通っていない。

**対処**:

```bash
which uv
uv --version
```

見つからない場合は uv の公式インストール手順に従う。

### 問題2: `Path to model ... not found.`

**原因**: checkpoint が `results/` にない。

**対処**:

```bash
ls -lh results/sam2.1_hiera_s/sam2.1_hiera_small.pt
```

存在しない場合はセクション 4.3 の checkpoint 取得を実行する。

### 問題3: CUDA capability / no kernel image エラー

**原因**: CUDA-enabled PyTorch wheel が現在の GPU 世代に対応していない。ONNX export は CPU で実行可能。

**対処**:

```bash
SAM2_BUILD_CUDA=0 uv run python tools/onnx_export.py --export image-encoder --encoder_type hiera_small --img_shape 3 1024 1024 --opset 17
```

`tools/onnx_export.py` では positional encoding の CUDA cache warmup を無効化して CPU export する。

### 問題4: simplified ONNX が大きくなる、または遅くなる

**原因**: constant folding により大きな tensor が graph に焼き込まれることがある。

**対処**:

```bash
--simplify-tensor-size-threshold 1MB
```

また、original と simplified の両方で実測してから採用する。

---

## 7. 次のステップ

### 7.1 実データ推論パイプラインの作成

* RGB image を image encoder ONNX に入力する前処理を定義する
* encoder の複数出力を decoder 入力へ mapping する
* BBOX / positive points / negative points から prompt embedding を生成する経路を整理する
* mask decoder ONNX の出力 mask logits を元画像解像度へ戻す

### 7.2 軽量 checkpoint の取得と export

* `nanosam2-resnet18`
* `nanosam2-mobilenetV3_large`

これらは対応 checkpoint が必要。checkpoint 配置先は `tools/onnx_export.py` の `ModelSource` を参照する。

### 7.3 ベンチマーク

* CPUExecutionProvider の速度
* GPU Execution Provider の可否
* original ONNX と simplified ONNX の比較
* 画像 1 枚あたりの encoder / decoder / postprocess 時間

---

## 8. 環境セットアップ完了チェックリスト

* [ ] `uv --version` が実行できる
* [ ] `git status --short --branch` で想定ブランチを確認した
* [ ] `uv sync` が完了した
* [ ] `torch`, `onnx`, `onnxruntime`, `onnxsim` を import できる
* [ ] `results/sam2.1_hiera_s/sam2.1_hiera_small.pt` が存在する
* [ ] image encoder ONNX を生成できる
* [ ] mask decoder ONNX を生成できる
* [ ] `onnx.checker` が通る
* [ ] ORT CPUExecutionProvider で load とダミー推論が通る
* [ ] `uv pip install` ではなく `uv add` / `uv sync` を使っている

---

## 9. 更新履歴

* `2026-06-30 12:30 JST` 初版作成。uv 環境、SAM2.1 Hiera small ONNX export、onnxsim、検証手順を記載。

---

## このドキュメントについて

本ガイドは、新しいセッションや新規参加メンバーが、短時間で同一の開発・実行環境を再現し、ONNX export 作業を継続できるようにすることを目的とする。
環境セットアップで問題が発生した場合は、本ドキュメントのトラブルシューティングに加え、`README.md`、`tools/onnx_export.py`、`pyproject.toml`、`uv.lock` を確認する。
