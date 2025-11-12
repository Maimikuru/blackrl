# 学習曲線のプロット方法

## 概要

SAC-IRLFアルゴリズムの訓練では、以下のメトリクスが自動的に記録されます：

1. **Leader Objective**: リーダーの目的関数の値
2. **IRL Gradient Norm**: MDCE IRLの勾配ノルム（収束の指標）
3. **IRL Likelihood**: MDCE IRLの尤度（エキスパート軌跡との一致度）
4. **Leader Gradient Norm**: リーダー方策の勾配ノルム
5. **Leader Mean Q-value**: リーダーのQ値の平均

## 使い方

### 方法1: 訓練時に自動生成

`train_bilevel.py`を実行すると、自動的にプロットが生成されます：

```bash
uv run scripts/train_bilevel.py
```

出力:
- `outputs/training_stats.pkl`: 統計データ（pickle形式）
- `outputs/learning_curves.png`: 学習曲線のプロット

### 方法2: 保存された統計から手動生成

既存の統計ファイルからプロットを生成：

```bash
python scripts/plot_learning_curves.py outputs/training_stats.pkl
```

### 方法3: Pythonスクリプト内で使用

```python
from scripts.plot_learning_curves import plot_learning_curves
import matplotlib.pyplot as plt

# 訓練を実行
algo, stats = main()

# プロットを生成
plot_learning_curves(stats, save_path="my_results.png")
plt.show()
```

## プロットの見方

### 1. Leader Objective (左上)
- リーダーの累積報酬
- **上昇傾向** = 学習が進んでいる
- 安定した値に収束すれば成功

### 2. MDCE IRL Gradient Norm (中上)
- 報酬推定の収束度
- **緑の破線** = 収束閾値（0.025）
- この線を下回れば収束

### 3. MDCE IRL Likelihood (右上)
- エキスパート軌跡との一致度
- **増加傾向** = 報酬推定が改善
- 値が高いほど良い

### 4. Leader Policy Gradient Norm (左下)
- リーダー方策の更新度
- 大きすぎると不安定、小さすぎると学習が遅い

### 5. Leader Mean Q-value (中下)
- リーダーのQ値推定
- 環境の報酬スケールに応じた値を期待

### 6. Training Summary (右下)
- 訓練の要約統計
- 最終的な性能指標

## トラブルシューティング

### プロットが生成されない

**エラー**: `ModuleNotFoundError: No module named 'matplotlib'`

**解決策**:
```bash
uv sync  # 依存関係を更新
```

### 統計データが空

**原因**: 訓練が早期に終了した、またはエラーが発生した

**解決策**:
- `verbose=True`で詳細ログを確認
- エラーメッセージをチェック

### プロットが表示されない（ヘッドレス環境）

**解決策**:
- `train_bilevel.py`の`plt.show()`をコメントアウト
- または、PNG画像を直接確認

## カスタマイズ

### 追加のメトリクスを記録

`bilevel_rl.py`の`train`メソッド内で：

```python
# カスタムメトリクスを追加
self.stats["my_custom_metric"].append(my_value)
```

### プロットのスタイル変更

`plot_learning_curves.py`を編集して、色、線のスタイル、フォントサイズなどを変更できます。

## 例

実行後、以下のような出力が得られます：

```
Training completed!
Final leader objective: 123.4567

Statistics saved to: outputs/training_stats.pkl
Learning curves saved to: outputs/learning_curves.png
```

生成されたPNG画像を開くと、6つのサブプロットで学習の進捗が視覚化されています。

