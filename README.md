# Movie-serch

[![CI](https://github.com/hirorogo/Movie-serch/actions/workflows/ci.yml/badge.svg)](https://github.com/hirorogo/Movie-serch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Private-gray.svg)](#ライセンス)

動画に出演している人物を AI で自動判定するツール。

## 概要

- 動画には最大 **3人** の出演者が登場するが、動画によっては **2人だけ** の場合もある
- 顔にはモザイクがかかっているため、**声紋分析を主軸**とし、視覚的特徴を補助に使う
- 複数の AI / モデルを組み合わせて判定精度を高める

### 主な特徴

| 特徴 | 説明 |
|------|------|
| 声紋ベースの話者識別 | Resemblyzer による d-vector 埋め込みで高精度な話者照合 |
| 自動話者分離 | pyannote-audio による発話区間の自動検出 |
| 視覚的補助分析 | YOLOv8 + OpenCLIP で体型・髪型・服装を補助判定 |
| インタラクティブセットアップ | 動画1本から基準音声を自動登録するウィザード |
| バッチ一括解析 | フォルダ内の全動画を自動解析、差分解析にも対応 |
| 中断復帰 | 解析途中で中断しても、既存の結果を引き継いで再開可能 |
| 埋め込みキャッシュ | 一度計算した声紋ベクトルをキャッシュして高速化 |

## 全自動ワークフロー — 動画を入れるだけ

このツールは「初回セットアップだけ済ませれば、あとは動画を放り込むだけ」で完結するように設計されています。

### 最短2ステップで完了

```
STEP 1（初回のみ）: 基準音声の自動登録
  → 出演者がわかる動画を1本食わせる
  → AI が声を自動分離 → 自動ラベリング → 「合ってる？(y)」で終了

STEP 2（以降はこれだけ）: 動画フォルダを指定して放置
  → フォルダに動画を入れてコマンド1つ叩くだけ
  → 全動画を順番に自動解析 → JSON/CSV で結果が出る
```

```bash
# STEP 1: 初回セットアップ（1回だけ）
python src/main.py setup --video 出演者がわかる動画.mp4

# STEP 2: あとは動画を入れてコマンドを叩くだけ
python src/main.py auto-analyze --dir /path/to/videos/
```

### 自動セットアップウィザード（`setup`）

基準音声の登録は手動で WAV ファイルを用意する必要はありません。
出演者がわかっている動画を1本指定するだけで、AI が全て自動で処理します。

```
============================================================
 セットアップウィザード
============================================================

対象動画: known_video.mp4

[Step 1/6] 音声を抽出中...
  動画の長さ: 12分34秒

[Step 2/6] 話者を自動分離中...                    ← AI が話者の切り替わりを自動検出
  3 人の話者を検出しました。

[Step 3/6] 各話者の音声サンプルを作成中...         ← 各話者の代表的な発話区間を自動切り出し
  話者 speaker_0: 15区間, 合計 52.3秒
  話者 speaker_1: 12区間, 合計 41.8秒
  話者 speaker_2: 8区間, 合計 28.1秒

[Step 4/6] 各話者の声の特徴を分析中...             ← ピッチ・声質を自動分析
  speaker_0: 声の高さ=高め (235Hz), 発話量=52.3秒 (15区間)
  speaker_1: 声の高さ=中間 (178Hz), 発話量=41.8秒 (12区間)
  speaker_2: 声の高さ=低め (125Hz), 発話量=28.1秒 (8区間)

[Step 5/6] AIが既存の基準音声と照合して自動判定中... ← AI が「この声は誰か」を自動判定
  ──────────────────────────────────────────────────
  AIの自動判定結果:
  ──────────────────────────────────────────────────
    speaker_0 → person_a (出演者A)
      類似度: 0.9234 [高]  声の高さ: 高め  発話量: 52.3秒
    speaker_1 → person_b (出演者B)
      類似度: 0.8891 [高]  声の高さ: 中間  発話量: 41.8秒
    speaker_2 → person_c (出演者C)
      類似度: 0.8456 [中]  声の高さ: 低め  発話量: 28.1秒

  操作を選択してください:
    y  → この割り当てで決定           ← 「y」を押すだけで完了
    e  → 手動で修正する
    q  → キャンセル

  [y/e/q]: y
  → 割り当てを確定しました。

[Step 6/6] 基準音声を保存中...
  保存: data/reference_voices/person_a/reference_0.wav (8.2秒)
  保存: data/reference_voices/person_a/reference_1.wav (6.5秒)
  ...

============================================================
 セットアップ完了!
============================================================
```

**ウィザードが自動でやること:**

| ステップ | 自動処理の内容 |
|----------|--------------|
| 音声抽出 | 動画から音声トラックを自動抽出（FFmpeg） |
| 話者分離 | AI が「誰がいつ話しているか」を時間区間で自動検出（pyannote-audio） |
| サンプル切り出し | 各話者の代表的な発話区間を自動で切り出して WAV 保存 |
| 声の特徴分析 | ピッチ（声の高さ）・発話量を自動で分析（librosa） |
| AI 自動ラベリング | 既存の基準音声がある → コサイン類似度で自動照合。ない → 声の特徴でソートして仮割当 |
| 基準音声保存 | 確定後、各出演者フォルダに基準音声を自動配置 |

> **初回（基準音声なし）の場合**: AI がピッチの高い順に出演者 A/B/C と仮割当して提案。間違っていれば `e` で修正可能。
> **2回目以降（基準音声あり）の場合**: 既存の声紋と照合して類似度ベースで自動判定するので、精度がさらに向上。

### 全自動バッチ解析（`auto-analyze`）

**基準音声の登録さえ済んでいれば、あとは動画ファイルをフォルダに入れてコマンドを叩くだけ。** 何も考えなくて OK。

```bash
python src/main.py auto-analyze --dir /path/to/videos/
```

```
動画 150 件を検出しました。
パイプラインを初期化中...
解析済み: 120 件（スキップ）           ← 前回解析済みの動画は自動スキップ
  [1/150] スキップ: video_001.mp4
  [2/150] スキップ: video_002.mp4
  ...
  [121/150] 解析中: video_121.mp4      ← 新しい動画だけを自動解析
  [122/150] 解析中: video_122.mp4
  ...
  [150/150] 解析中: video_150.mp4

新規解析: 30 件 / 合計: 150 件
結果を保存しました: output/results.json
結果を保存しました: output/results.csv
```

**auto-analyze が自動でやること:**

| 機能 | 説明 |
|------|------|
| 動画の自動検出 | フォルダ内の動画ファイル（mp4, avi, mkv, mov, wmv, flv, webm）を自動で全件検出 |
| 解析済みスキップ | `output/results.json` を参照し、既に解析済みの動画を自動スキップ。新しい動画だけ処理する |
| 再帰検索 | `--recursive` オプションでサブフォルダ内も自動走査 |
| 逐次保存（中断復帰） | 1動画の解析が終わるたびに結果を自動保存。途中で中断しても次回コマンドを叩けば続きから再開 |
| 一時ファイル自動削除 | 中間生成ファイル（一時 WAV、一時フレーム画像等）を処理後に自動クリーンアップ |
| JSON + CSV 出力 | `--format both` で機械読み取り用 JSON とスプレッドシート用 CSV を同時生成 |

### 日常運用イメージ

```
【初回セットアップ済みの状態】

① 新しい動画を入手
② /path/to/videos/ フォルダに放り込む
③ ターミナルで以下を実行:

   python src/main.py auto-analyze --dir /path/to/videos/

④ 放置（動画1本あたり数分で処理）
⑤ output/results.json に結果が出力される
⑥ 次に新しい動画が来たら ②〜⑤ を繰り返すだけ
```

### その他の自動化機能

#### 自動フォールバック

pyannote-audio のセットアップ（HuggingFace トークン等）が完了していなくても動作します。
pyannote が使えない場合は、Resemblyzer ベースの簡易話者分離に自動的にフォールバックします。

```
pyannote-audio の初期化に失敗。resemblyzer にフォールバックします
resemblyzer でダイアライゼーション実行中（フォールバック）...
```

> pyannote-audio の方が高精度ですが、Resemblyzer 単体でも基本的な話者分離は可能です。

#### 埋め込みキャッシュの自動管理

一度計算した声紋ベクトル（256次元 d-vector）は `.cache/embeddings/` に自動で `.npy` 形式でキャッシュされます。
同じ音声ファイルを再度解析する際はキャッシュから読み込まれるため、計算をスキップして高速に処理されます。

- キャッシュキーはファイルの **MD5 ハッシュ** なので、ファイル名を変更しても同一ファイルは再計算されない
- キャッシュのヒット/ミス統計を内部で追跡
- 手動でクリアする場合: `.cache/embeddings/` を削除

#### システム要件の自動チェック（preflight）

解析実行前に、必要なシステム要件が揃っているかを自動でチェックします。
足りないものがあればエラーメッセージで何が必要かを教えてくれます。

- FFmpeg / FFprobe がインストールされているか
- GPU が利用可能か（`--visual` 使用時）
- 基準音声が登録されているか

## 出演者定義

| ID | 呼称（仮） | 備考 |
|----|-----------|------|
| A  | 未設定     | 基準音声を登録して使用 |
| B  | 未設定     | 基準音声を登録して使用 |
| C  | 未設定     | 基準音声を登録して使用 |

> 呼称は利用者が自由に設定できる。
> 各出演者の基準音声サンプル（10〜30秒程度）を `data/reference_voices/` に配置する。

## 分析パイプライン

```
動画ファイル
  │
  ├─ [Step 1] 音声抽出（FFmpeg）
  │     └─ WAV ファイル生成（16kHz モノラル）
  │
  ├─ [Step 2] 話者ダイアライゼーション（pyannote-audio）
  │     └─ 「誰がいつ話しているか」を時間区間で分離
  │
  ├─ [Step 3] 声紋照合（Resemblyzer）
  │     └─ 各区間の音声 ↔ 基準音声の類似度スコア算出
  │
  ├─ [Step 4] 視覚的特徴分析（補助・オプション）
  │     ├─ 体型・シルエット検出（YOLOv8）
  │     ├─ 髪型・髪色の判定（OpenCLIP）
  │     └─ 服装・アクセサリーの検出
  │
  └─ [Step 5] 総合判定
        ├─ 声紋スコア（主: 70%）+ 視覚スコア（副: 30%）を統合
        ├─ 閾値に基づき出演有無を判定
        └─ 結果出力（JSON / CSV）
```

### データフロー詳細

```
入力: video.mp4（顔モザイク・音声あり）
  ↓
[FFmpeg] 音声抽出 → audio.wav (16kHz mono)
  ↓
[pyannote-audio] 話者ダイアライゼーション
  → segments: [(0:00-0:15, speaker_0), (0:15-0:30, speaker_1), ...]
  ↓
[Resemblyzer] 各セグメントの声紋埋め込みベクトル生成
  ↓
[VoiceMatcher] 基準音声との コサイン類似度 算出
  → person_a=0.92, person_b=0.87, person_c=0.12
  ↓
[Optional: YOLOv8 + OpenCLIP] フレーム抽出 → 人物検出 → 視覚特徴比較
  ↓
[Pipeline] スコア統合: voice_score × 0.7 + visual_score × 0.3
  ↓
出力: JSON / CSV レポート
```

## 使用する AI / モデル

| 役割 | モデル / ライブラリ | 説明 |
|------|-------------------|------|
| 声紋埋め込み | [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | 音声から話者の特徴ベクトル（d-vector）を生成 |
| 話者分離 | [pyannote-audio](https://github.com/pyannote/pyannote-audio) | 音声内の話者交代を時間区間で検出 |
| 視覚分析 | [YOLOv8](https://github.com/ultralytics/ultralytics) | 人物検出・体型推定 |
| 特徴抽出 | [OpenCLIP](https://github.com/mlfoundations/open_clip) | 画像から視覚的特徴ベクトルを抽出 |
| 音声処理 | [librosa](https://github.com/librosa/librosa) | ピッチ分析・音声前処理 |
| 総合判定 | カスタムロジック | 各 AI の出力スコアを統合して最終判定 |

> モデルは段階的に追加する。まず声紋系（Resemblyzer + pyannote）から始める。

## ディレクトリ構成

```
Movie-serch/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml                  # 閾値・モデル設定
│
├── src/
│   ├── __init__.py
│   ├── main.py                  # CLI エントリーポイント
│   ├── pipeline.py              # 分析パイプライン統合
│   ├── preflight.py             # システム要件チェック
│   ├── setup_wizard.py          # 基準音声セットアップウィザード
│   ├── cache.py                 # 埋め込みベクトルのキャッシュ
│   │
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── extractor.py         # 動画 → 音声抽出（FFmpeg）
│   │   ├── diarizer.py          # 話者ダイアライゼーション
│   │   └── voice_matcher.py     # 声紋照合
│   │
│   ├── visual/
│   │   ├── __init__.py
│   │   ├── frame_extractor.py   # 動画 → フレーム画像抽出
│   │   ├── body_analyzer.py     # 体型・シルエット分析
│   │   └── appearance.py        # 髪型・服装分析
│   │
│   └── output/
│       ├── __init__.py
│       └── reporter.py          # 結果出力（JSON / CSV）
│
├── data/
│   ├── reference_voices/        # 基準音声サンプル
│   │   ├── person_a/
│   │   ├── person_b/
│   │   └── person_c/
│   ├── reference_visuals/       # 基準画像サンプル（任意）
│   │   ├── person_a/
│   │   ├── person_b/
│   │   └── person_c/
│   └── videos/                  # 解析対象の動画ファイル
│
├── output/                      # 解析結果の出力先
│
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI
│
└── tests/
    ├── __init__.py
    ├── test_extractor.py
    ├── test_voice_matcher.py
    ├── test_pipeline.py
    ├── test_cache.py
    ├── test_preflight.py
    └── test_visual.py
```

## セットアップ

### 前提条件

- Python 3.10 以上
- FFmpeg がインストールされていること
- GPU 推奨（pyannote-audio / YOLO の高速化）
- [HuggingFace アカウント](https://huggingface.co/)（pyannote-audio のモデルダウンロードに必要）

### FFmpeg のインストール

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg

# インストール確認
ffmpeg -version
```

### プロジェクトのインストール

```bash
git clone https://github.com/hirorogo/Movie-serch.git
cd Movie-serch
pip install -r requirements.txt
```

### HuggingFace トークンの設定

pyannote-audio による話者ダイアライゼーションを使用するには、HuggingFace トークンが必要です。

1. [HuggingFace](https://huggingface.co/settings/tokens) でアクセストークンを取得
2. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) の利用規約に同意
3. 環境変数またはコマンドライン引数で設定

```bash
# 環境変数で設定（推奨）
export HF_TOKEN="hf_your_token_here"

# または、コマンド実行時に指定
python src/main.py analyze --video video.mp4 --hf-token "hf_your_token_here"
```

### 基準音声の準備（自動セットアップ）

出演者がわかっている動画を1本用意して、セットアップコマンドを実行するだけ。
音声の切り出しは自動で行われ、「この声は誰？」と聞かれるので番号で答えるだけ。

```bash
python src/main.py setup --video 出演者がわかる動画.mp4
```

```
[Step 1/5] 音声を抽出中...
[Step 2/5] 話者を自動分離中...
  3 人の話者を検出しました。
[Step 3/5] 各話者の音声サンプルを作成中...
[Step 4/5] 話者のラベル付け

  --- 話者クラスタ: speaker_0 ---
  発話区間: 12区間, 合計 45.3秒
  この話者は誰ですか？
    1. person_a (出演者A)
    2. person_b (出演者B)
    3. person_c (出演者C)
  選択 [1-3/s/q]: 1
  → speaker_0 = person_a (出演者A)
  ...

[Step 5/5] 基準音声を保存中...
セットアップ完了!
```

> 手動で配置する場合は `data/reference_voices/person_a/sample.wav` のように直接配置も可能。

## 使い方

### 設計思想 — 手作業ゼロを目指す

従来のアプローチでは、基準音声の準備にかなりの手作業が必要でした。

**Before（従来の面倒な方法）:**
```
① 各出演者の音声サンプルを手動で切り出す
② 各出演者の画像を手動で切り出す
③ data/reference_voices/person_a/ に手動で配置する
④ やっと解析開始
```

**After（現在の全自動フロー）:**
```
① python src/main.py setup --video 出演者がわかる動画.mp4
  → AI が自動で話者分離 → 自動ラベリング → 「y」で確定
② python src/main.py auto-analyze --dir 動画フォルダ/
  → 動画を入れるだけ、あとは全自動
```

> ユーザーが手動で音声を切り分ける必要は一切ありません。
> AI が話者を自動検出し、声紋の照合まで全て自動で行います。

### CLI コマンド一覧

| コマンド | 説明 |
|----------|------|
| `analyze` | 単一動画またはフォルダを解析して出演者を判定 |
| `setup` | 基準音声のインタラクティブセットアップ |
| `auto-analyze` | フォルダ内の全動画を自動バッチ解析 |
| `list-speakers` | 登録済みの基準音声を一覧表示 |
| `test-voice` | 声紋照合のテスト実行（デバッグ用） |

### グローバルオプション

```bash
python src/main.py --verbose [コマンド]   # デバッグログを表示
python src/main.py --help                 # ヘルプを表示
```

### 単一動画の解析

```bash
python src/main.py analyze --video path/to/video.mp4
```

### フォルダ一括解析

```bash
python src/main.py analyze --dir path/to/videos/
```

### 自動バッチ解析（推奨）

大量の動画をまとめて解析する場合は `auto-analyze` を使用します。
解析済みの動画は自動でスキップされるため、新しい動画を追加して再実行するだけで差分解析が可能です。

```bash
# 基本的な使い方
python src/main.py auto-analyze --dir /path/to/videos/

# サブフォルダも再帰的に検索
python src/main.py auto-analyze --dir /path/to/videos/ --recursive

# 視覚分析も有効にして JSON + CSV 両方を出力
python src/main.py auto-analyze --dir /path/to/videos/ --visual --format both
```

### 視覚分析を有効にする

デフォルトでは声紋分析のみが有効です。視覚分析も併用するには `--visual` フラグを追加します。

```bash
python src/main.py analyze --video video.mp4 --visual
```

### 声紋照合のテスト

基準音声の登録状態や照合精度を確認するためのデバッグコマンドです。

```bash
python src/main.py test-voice --video test.mp4
```

```
声紋モデルを読み込み中...
登録話者: ['person_a', 'person_b', 'person_c']

音声を抽出中: test.mp4
声紋照合中...

声紋類似度スコア:
  ○ person_a: 0.9234
  ○ person_b: 0.8712
  × person_c: 0.1245
```

### 登録済み話者の確認

```bash
python src/main.py list-speakers
```

### 出力例

```json
{
  "video": "example.mp4",
  "duration": "12:34",
  "performers": {
    "person_a": {
      "detected": true,
      "confidence": 0.92,
      "speaking_time": "3:21"
    },
    "person_b": {
      "detected": true,
      "confidence": 0.87,
      "speaking_time": "4:05"
    },
    "person_c": {
      "detected": false,
      "confidence": 0.12,
      "speaking_time": "0:00"
    }
  },
  "summary": "出演者: person_a, person_b（2名）"
}
```

## 設定ファイル（config.yaml）

```yaml
performers:
  - id: person_a
    name: "出演者A"
  - id: person_b
    name: "出演者B"
  - id: person_c
    name: "出演者C"

paths:
  reference_voices: "data/reference_voices"
  reference_visuals: "data/reference_visuals"
  videos: "data/videos"
  output: "output"

thresholds:
  voice_similarity: 0.75       # 声紋一致と判定する最低スコア
  visual_similarity: 0.60      # 視覚一致と判定する最低スコア
  combined_weight_voice: 0.7   # 総合判定での声紋の重み
  combined_weight_visual: 0.3  # 総合判定での視覚の重み

diarization:
  min_segment_duration: 1.0    # 最短発話区間（秒）
  max_speakers: 3              # 最大話者数

audio:
  sample_rate: 16000           # サンプリングレート（Hz）
  extract_format: "wav"        # 抽出形式

visual:
  frame_interval: 2.0          # フレーム抽出間隔（秒）
  confidence_threshold: 0.5    # YOLO 人物検出の信頼度閾値

output:
  format: "json"               # json / csv / both
```

### 設定のカスタマイズ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `thresholds.voice_similarity` | 0.75 | 声紋が一致と判定する最低コサイン類似度。高くすると厳密、低くすると寛容になる |
| `thresholds.visual_similarity` | 0.60 | 視覚的特徴が一致と判定する最低スコア |
| `thresholds.combined_weight_voice` | 0.7 | 総合判定における声紋スコアの重み（0.0〜1.0） |
| `thresholds.combined_weight_visual` | 0.3 | 総合判定における視覚スコアの重み（0.0〜1.0） |
| `diarization.min_segment_duration` | 1.0 | 最短発話区間（秒）。短い区間はノイズとして除外 |
| `diarization.max_speakers` | 3 | 最大話者数 |
| `audio.sample_rate` | 16000 | 音声抽出時のサンプリングレート（Hz） |
| `visual.frame_interval` | 2.0 | フレーム抽出間隔（秒）。短くすると精度向上するが処理時間が増加 |

## 対応動画形式

| 拡張子 | 形式 |
|--------|------|
| `.mp4` | MPEG-4 |
| `.avi` | AVI |
| `.mkv` | Matroska |
| `.mov` | QuickTime |
| `.wmv` | Windows Media Video |
| `.flv` | Flash Video |
| `.webm` | WebM |

## 開発

### テストの実行

```bash
# 全テスト実行
python -m pytest tests/ -v

# 特定のテストファイル
python -m pytest tests/test_voice_matcher.py -v

# カバレッジ付き
python -m pytest tests/ -v --tb=short
```

### CI/CD

GitHub Actions による自動テストが設定されています。

- **トリガー**: `main` ブランチへの push / pull request
- **環境**: Ubuntu (Python 3.10, 3.11, 3.12)
- **内容**: FFmpeg インストール → 依存関係インストール → pytest 実行

### コード構成

| モジュール | 責務 |
|-----------|------|
| `src/main.py` | CLI エントリーポイント。Click によるコマンド定義 |
| `src/pipeline.py` | 分析パイプライン全体の統合・オーケストレーション |
| `src/preflight.py` | FFmpeg・GPU 等のシステム要件チェック |
| `src/setup_wizard.py` | 基準音声のインタラクティブ登録ウィザード |
| `src/cache.py` | 声紋埋め込みベクトルの .npy キャッシュ管理 |
| `src/audio/extractor.py` | FFmpeg を使った動画→音声抽出 |
| `src/audio/diarizer.py` | pyannote-audio / フォールバック話者分離 |
| `src/audio/voice_matcher.py` | Resemblyzer による声紋照合・スコア算出 |
| `src/visual/frame_extractor.py` | FFmpeg を使ったフレーム画像抽出 |
| `src/visual/body_analyzer.py` | YOLOv8 による人物検出・体型分析 |
| `src/visual/appearance.py` | OpenCLIP による髪型・服装の特徴抽出 |
| `src/output/reporter.py` | JSON / CSV レポートの生成・保存 |

## 開発ロードマップ

### Phase 1: 基盤構築
- [x] README 作成
- [x] プロジェクト構造・依存関係の定義
- [x] 動画 → 音声抽出モジュール
- [x] システム要件チェック（preflight）
- [x] CI/CD パイプライン構築

### Phase 2: 声紋分析（主軸）
- [x] 基準音声の登録・声紋ベクトル生成
- [x] 話者ダイアライゼーション（区間分離）
- [x] 声紋照合による出演者判定
- [x] 2人のみ出演のケースへの対応
- [x] セットアップウィザード
- [x] 埋め込みキャッシュ

### Phase 3: 視覚分析（補助）
- [x] フレーム画像抽出
- [x] 人物検出（YOLO）
- [x] 体型・髪型・服装の特徴抽出
- [x] 視覚的特徴による補助判定

### Phase 4: 統合・出力
- [x] 声紋 + 視覚スコアの統合判定ロジック
- [x] JSON / CSV レポート出力
- [x] CLI インターフェース完成
- [x] 自動バッチ解析（auto-analyze）

### Phase 5: 精度向上
- [ ] 閾値チューニング
- [ ] テストデータでの検証
- [ ] エッジケース対応（BGM が大きい、複数人同時発話 等）

## 開発の経緯

このツールは段階的に自動化を進めてきました。

### v1: 手動切り出し時代

最初の設計では、ユーザーが自分で各出演者の音声サンプルと画像を手動で切り出し、
所定のフォルダに配置する必要がありました。

```
基盤
 * requirements.txt — 全依存ライブラリ
 * .gitignore — 動画・モデルファイル等の除外
 * config.yaml — 閾値・パス・出演者定義

音声分析（主軸）
 * src/audio/extractor.py — FFmpegで動画→WAV抽出（全体・区間指定対応）
 * src/audio/voice_matcher.py — Resemblyzerで声紋ベクトル生成・照合・出演者判定
 * src/audio/diarizer.py — 話者分離（pyannote優先、resemblyzerフォールバック）

視覚分析（補助）
 * src/visual/frame_extractor.py — 動画→フレーム画像抽出
 * src/visual/body_analyzer.py — YOLOv8で人物検出・体型分析
 * src/visual/appearance.py — OpenCLIPで外見特徴ベクトル照合

統合・出力
 * src/pipeline.py — 声紋+視覚スコアの統合判定パイプライン（2人のみ出演にも対応）
 * src/output/reporter.py — JSON/CSV出力・出演マトリクス表示
 * src/main.py — CLIツール（analyze, list-speakers, test-voice コマンド）
```

**課題**: 各出演者の音声を手動で切り出す必要があり、かなり面倒。

### v2: setup コマンドの追加（半自動ラベリング）

「自分で人を切り分ける必要があるのか？」という問題を解決するため、
`setup` コマンドを追加。動画1本を食わせるだけで話者を自動分離し、
ユーザーは「この声は誰？」と聞かれて番号で答えるだけに。

```
python src/main.py setup --video 出演者がわかる動画.mp4
→ 自動で話者を3つに分離
→ 「クラスタ1の声を再生します。誰ですか？ [A/B/C]」
→ ユーザーがラベルを付けるだけ
→ 基準データが自動保存される
```

**追加ファイル**: `src/setup_wizard.py`

### v3: AI 自動ラベリング（現在）

「ユーザーが聞き分けられない場合は？」という問題を解決するため、
AI が自動で話者のラベリングまで行い、ユーザーは確認するだけの仕組みに進化。

- **基準音声がある場合** → 既存の声紋とコサイン類似度で自動照合
- **初回（基準なし）** → 声の特徴（ピッチ・発話量）を分析して自動でソート・仮割当

さらに `auto-analyze` コマンドも追加し、フォルダに動画を放り込んで
コマンドを叩くだけの完全自動ワークフローを実現。

**追加機能**: AI 自動ラベリング、`auto-analyze` コマンド、差分解析、中断復帰、埋め込みキャッシュ

## トラブルシューティング

### FFmpeg が見つからない

```
エラー: FFmpeg がインストールされていません
```

FFmpeg がシステムの PATH に存在するか確認してください。

```bash
ffmpeg -version   # バージョンが表示されれば OK
```

### HuggingFace トークンエラー

```
エラー: HuggingFace トークンが無効です
```

- トークンが正しく設定されているか確認: `echo $HF_TOKEN`
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) のモデルページで利用規約に同意しているか確認

### 基準音声が登録されていない

```
エラー: 基準音声が登録されていません
```

`setup` コマンドで基準音声を登録するか、`data/reference_voices/person_a/` 等に WAV ファイルを手動配置してください。

### GPU が認識されない

pyannote-audio や YOLO は CPU でも動作しますが、処理速度が大幅に低下します。
GPU を使用する場合は、CUDA 対応の PyTorch がインストールされているか確認してください。

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 声紋スコアが低い

- 基準音声のサンプルが短すぎる可能性があります（10〜30秒推奨）
- BGM や環境音が大きい場合、精度が低下します
- `config.yaml` の `voice_similarity` 閾値を調整してみてください

## ライセンス

このプロジェクトは個人利用を目的としています。
