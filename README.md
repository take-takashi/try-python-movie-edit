# try-python-movie-edit

pythonでmp4動画編集をやってみる

## 環境の確認

```bash
try-python-movie-edit % pyenv -v
pyenv 2.5.2
try-python-movie-edit % poetry -V
Poetry (version 2.0.1)
try-python-movie-edit % poetry config virtualenvs.in-project
true

# trueであることを確認した
```

## python実行環境の作成

```bash
# python3.13.1をインストール（すでにインストール済みだった）
try-python-movie-edit % pyenv install 3.13.1
pyenv: /Users/takashi/.pyenv/versions/3.13.1 already exists
continue with installation? (y/N) N

# ローカル環境でpython3.13.1を使用する
% pyenv local 3.13.1

# poetryでプロジェクトを作成
% poetry init
% poetry install --no-root

# shellを実行するためにプラグインを導入（これで良かったのか？）
% poetry self add poetry-plugin-shell

% poetry shell
```

## ジュピターの設定

```bash
% poetry add -D ipykernel
% poetry add opencv-python
% poetry add numpy
```

## 高速化

```bash
% poetry add torch

# numpyのバージョンを下げるため（ultralyticsの依存関係）
% poetry remove numpy
% poetry add numpy==2.1.1
% poetry add ultralytics
% poetry add omegaconf

# 環境変数を読み込むため
% poetry add python-dotenv
```
