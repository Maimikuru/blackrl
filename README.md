# blackrl
卒研の実験リポジトリ

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

bilevel-rl（二層強化学習）

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
└── src   <- このプロジェクトで使用するソースコード
    │
    ├── __init__.py             <- blackrlをPythonモジュールにする
    │
    ├── config.py               <- 有用な変数と設定を格納
    │
    ├── dataset.py              <- データをダウンロードまたは生成するスクリプト
    │
    ├── features.py             <- モデリング用の特徴量を作成するコード
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- 訓練済みモデルでモデル推論を実行するコード          
    │   └── train.py            <- モデルを訓練するコード
    │
    └── plots.py                <- 可視化を作成するコード
```

--------

