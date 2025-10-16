# クイックスタートガイド / Quick Start Guide

## インストール手順

### 1. uvをインストール

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**または pip経由:**
```bash
pip install uv
```

### 2. リポジトリのクローンとセットアップ

```bash
git clone https://github.com/Maimikuru/blackrl.git
cd blackrl

# 依存関係をインストール
uv sync

# 開発用ツール（pytest, black, ruff, jupyter等）もインストール
uv sync --all-extras
```

## 使い方

### サンプルスクリプトの実行

```bash
# ランダムエージェントのサンプル
uv run python examples/cartpole_random.py

# エージェントクラスを使用したサンプル
uv run python examples/agent_example.py
```

### Pythonシェルで使う

```bash
uv run python
```

```python
>>> import gymnasium as gym
>>> from blackrl import RandomAgent
>>> env = gym.make("CartPole-v1")
>>> agent = RandomAgent(env.observation_space, env.action_space)
>>> observation, info = env.reset()
>>> action = agent.select_action(observation)
>>> print(action)
```

### Jupyterノートブックで実験

```bash
uv run jupyter notebook
```

## テストの実行

```bash
# すべてのテストを実行
uv run pytest tests/

# 詳細な出力
uv run pytest tests/ -v

# 特定のテストファイルを実行
uv run pytest tests/test_agents.py
```

## コードの品質チェック

```bash
# コードフォーマット
uv run black src/ tests/ examples/

# リントチェック
uv run ruff check src/ tests/ examples/

# リント問題を自動修正
uv run ruff check --fix src/ tests/ examples/
```

## 新しいエージェントの作成

`BaseAgent`を継承して独自のエージェントを作成できます：

```python
from blackrl import BaseAgent

class MyAgent(BaseAgent):
    def select_action(self, observation):
        # アクションの選択ロジック
        return self.action_space.sample()
    
    def update(self, observation, action, reward, next_observation, done):
        # 学習ロジック
        return {"loss": 0.0}
```

## Codonの使用（オプション）

高速化が必要な場合、Codonを使用できます：

### Codonのインストール

```bash
# Linux/macOS
/bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"
```

### Codonでコンパイル

```bash
# スクリプトをコンパイル
codon build -release your_script.py

# 実行
./your_script
```

## よくある環境（Environments）

Gymnasiumで利用可能な強化学習環境：

- **CartPole-v1**: 倒立振子の制御
- **MountainCar-v0**: 山を登る車の制御
- **LunarLander-v2**: 月面着陸船の制御
- **Atari games**: 各種Atariゲーム

```python
import gymnasium as gym

# 環境を作成
env = gym.make("CartPole-v1")

# 利用可能な環境を確認
from gymnasium import envs
print(envs.registry.keys())
```

## トラブルシューティング

### 依存関係のエラー

```bash
# 仮想環境を削除して再作成
rm -rf .venv
uv sync --all-extras
```

### PyTorchのGPU対応

デフォルトではCPU版のPyTorchがインストールされます。GPU版が必要な場合：

```bash
# CUDA対応のPyTorchをインストール（例: CUDA 11.8）
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 参考リンク

- [uv documentation](https://github.com/astral-sh/uv)
- [Gymnasium documentation](https://gymnasium.farama.org/)
- [PyTorch documentation](https://pytorch.org/)
- [Codon documentation](https://docs.exaloop.io/)
