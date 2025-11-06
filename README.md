# blackrl
卒研の実験リポジトリ

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

bilevel-rl（二層強化学習）

## セットアップ

```bash
# 仮想環境の作成（uvを使用）
make create_environment
source .venv/bin/activate  # Linux/macOS
# または
.\.venv\Scripts\activate  # Windows

# 依存関係のインストール
make requirements
# または
uv sync
```

## 実行方法

### 1. 基本的な使用例

```bash
# 各コンポーネントの使用例を実行
make example
# または
python scripts/example_usage.py
```

### 2. Bi-level RLの学習

```bash
# Bi-level RLの学習を実行
make train
# または
python scripts/train_bilevel.py
```

### 3. Pythonから直接インポート

```python
from blackrl.src.envs import DiscreteToyEnv1_1a
from blackrl.src.algos import BilevelRL
from blackrl.src.agents.follower import MDCEIRL, SoftQLearning

# 環境の作成
env = DiscreteToyEnv1_1a()

# Bi-level RLアルゴリズムの使用
# ... (詳細は scripts/example_usage.py を参照)
```

### 4. Jupyter Notebook

`notebooks/`ディレクトリにJupyterノートブックがあります。

```bash
jupyter notebook notebooks/
```

## プロジェクト構成

```
├── LICENSE            <- オープンソースライセンス（選択した場合）
├── Makefile           <- `make data` や `make train` などの便利なコマンドを含むMakefile
├── README.md          <- このプロジェクトを使用する開発者向けのトップレベルREADME
├── data
│   ├── external       <- 第三者ソースからのデータ
│   ├── interim        <- 変換された中間データ
│   ├── processed      <- モデリング用の最終的な標準データセット
│   └── raw            <- 元の変更不可なデータダンプ
│
├── docs               <- デフォルトのmkdocsプロジェクト。詳細はwww.mkdocs.orgを参照
│
├── models             <- 訓練済みおよびシリアル化されたモデル、モデル予測、またはモデル要約
│
├── notebooks          <- Jupyterノートブック。命名規則は番号（順序用）、
│                         作成者のイニシャル、短い`-`区切りの説明、例：
│                         `1.0-jqp-initial-data-exploration`
│
├── pyproject.toml     <- blackrlのパッケージメタデータとblackなどのツール設定を含む
│                         プロジェクト設定ファイル
│
├── references         <- データ辞書、マニュアル、その他の説明資料
│
├── reports            <- HTML、PDF、LaTeXなどで生成された分析結果
│   └── figures        <- レポートで使用する生成されたグラフィックと図
│
├── scripts            <- 実行可能なスクリプト
│   ├── example_usage.py <- 各コンポーネントの使用例
│   └── train_bilevel.py <- Bi-level RLの学習スクリプト
│
└── src   <- このプロジェクトで使用するソースコード
    │
    ├── __init__.py             <- blackrlをPythonモジュールにする
    │
    ├── config.py               <- 有用な変数と設定を格納
    │
    ├── envs                    <- Bi-level RL環境モジュール
    │   ├── __init__.py
    │   ├── base.py             <- 環境基底クラス（Environment, EnvSpec, GlobalEnvSpec）
    │   └── discrete_toy_env.py <- 離散環境の実装
    │
    ├── agents                  <- エージェントモジュール
    │   ├── follower
    │   │   ├── __init__.py
    │   │   ├── mdce_irl.py     <- MDCE IRL実装（報酬パラメータ推定）
    │   │   └── soft_q_learning.py <- Soft Q-Learning実装（フォロワー方策導出）
    │   └── leader              <- リーダーエージェントモジュール（将来の拡張用）
    │       └── __init__.py
    │
    ├── algos                   <- アルゴリズムモジュール
    │   ├── __init__.py
    │   └── bilevel_rl.py       <- Bi-level RLアルゴリズム
    │
    ├── policies                <- 方策モジュール
    │   ├── __init__.py
    │   └── joint_policy.py     <- リーダー・フォロワー統合方策
    │
    ├── q_functions             <- Q関数モジュール
    │   ├── __init__.py
    │   └── base.py             <- 表形式Q関数（TabularQFunction）
    │
    ├── replay_buffer           <- リプレイバッファモジュール（将来の拡張用）
    │   ├── __init__.py
    │   ├── base.py
    │   └── gamma_replay_buffer.py
    │
    └── plots.py                <- 可視化を作成するコード
```

## Bi-level強化学習の実装

このプロジェクトでは、Bi-level強化学習問題を解くための実装を提供しています。

### 主要コンポーネント

1. **環境 (envs/)**: 2人マルコフゲーム環境の基底クラス
   - `DiscreteToyEnv*`: 離散環境のバリエーション
2. **MDCE IRL (agents/follower/mdce_irl.py)**: フォロワーの報酬パラメータを推定
   - デモンストレーション軌跡から直接FEVを計算
3. **Soft Q-Learning (agents/follower/soft_q_learning.py)**: フォロワーの最適Max-Ent方策を導出
   - 表形式Qテーブルを使用（離散環境用）
4. **Bi-level RL (algos/bilevel_rl.py)**: リーダー・フォロワーのBi-level最適化
   - リーダーは現在`algos/`で管理（ptiaと同様の構造）
   - 将来の拡張用に`agents/leader/`ディレクトリを用意
5. **Q関数 (q_functions/)**: 表形式Q関数（TabularQFunction）
6. **Replay Buffer (replay_buffer/)**: 将来の拡張用（現在は未使用）

### 注意事項

- **リプレイバッファについて**: 現在の実装では、MDCE IRLとリーダーのQ関数計算はデモンストレーションデータを直接使用するため、リプレイバッファは使用していません。リプレイバッファは将来のオフポリシー学習拡張用に実装されています。

詳細な使用方法は `docs/bilevel_rl.md` を参照してください。

--------
