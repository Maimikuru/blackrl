# blackrl
卒研の実験リポジトリ / Graduation Research Experiment Repository

強化学習の実験を行うためのリポジトリです。

## 環境構築 / Environment Setup

このプロジェクトは [uv](https://github.com/astral-sh/uv) を使用してPython環境を管理します。

### uvのインストール / Installing uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### プロジェクトのセットアップ / Project Setup

```bash
# リポジトリをクローン
git clone https://github.com/Maimikuru/blackrl.git
cd blackrl

# 依存関係をインストール
uv sync

# 開発用の依存関係も含めてインストール
uv sync --all-extras
```

### 使い方 / Usage

```bash
# サンプルスクリプトを実行
uv run python examples/cartpole_random.py

# Pythonシェルを起動
uv run python

# Jupyterを起動
uv run jupyter notebook
```

## プロジェクト構成 / Project Structure

```
blackrl/
├── src/blackrl/     # メインパッケージ
├── examples/        # サンプルコード
├── tests/           # テストコード
└── pyproject.toml   # プロジェクト設定
```

## 依存関係 / Dependencies

主な依存関係:
- **gymnasium**: 強化学習環境
- **numpy**: 数値計算
- **torch**: ディープラーニング
- **matplotlib**: 可視化
- **tensorboard**: 学習の可視化

## Codonについて / About Codon

[Codon](https://github.com/exaloop/codon) は高性能なPythonコンパイラです。
必要に応じて以下のようにインストールできます:

```bash
# Codonのインストール (Linux/macOS)
/bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"
```

Codonを使用する場合は、`.py`ファイルを`.codon`としてコンパイルできます:
```bash
codon build -release your_script.py
```

## ライセンス / License

MIT
