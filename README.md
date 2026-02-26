# Movie-serch

動画に出演している人物を AI で自動判定するツール。

## 概要

- 動画には最大 **3人** の出演者が登場するが、動画によっては **2人だけ** の場合もある
- 顔にはモザイクがかかっているため、**声紋分析を主軸**とし、視覚的特徴を補助に使う
- 複数の AI / モデルを組み合わせて判定精度を高める

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
  │     └─ WAV ファイル生成
  │
  ├─ [Step 2] 話者ダイアライゼーション（pyannote-audio）
  │     └─ 「誰がいつ話しているか」を時間区間で分離
  │
  ├─ [Step 3] 声紋照合（resemblyzer）
  │     └─ 各区間の音声 ↔ 基準音声の類似度スコア算出
  │
  ├─ [Step 4] 視覚的特徴分析（補助）
  │     ├─ 体型・シルエット検出
  │     ├─ 髪型・髪色の判定
  │     └─ 服装・アクセサリーの検出
  │
  └─ [Step 5] 総合判定
        ├─ 声紋スコア（主）+ 視覚スコア（副）を統合
        ├─ 閾値に基づき出演有無を判定
        └─ 結果出力（JSON / CSV）
```

## 使用する AI / モデル

| 役割 | モデル / ライブラリ | 説明 |
|------|-------------------|------|
| 声紋埋め込み | [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | 音声から話者の特徴ベクトル（d-vector）を生成 |
| 話者分離 | [pyannote-audio](https://github.com/pyannote/pyannote-audio) | 音声内の話者交代を時間区間で検出 |
| 視覚分析 | [YOLOv8](https://github.com/ultralytics/ultralytics) | 人物検出・体型推定 |
| 特徴抽出 | [OpenCLIP](https://github.com/mlfoundations/open_clip) | 画像から視覚的特徴ベクトルを抽出 |
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
└── tests/
    ├── test_extractor.py
    ├── test_voice_matcher.py
    └── test_pipeline.py
```

## セットアップ

### 前提条件

- Python 3.10 以上
- FFmpeg がインストールされていること
- GPU 推奨（pyannote-audio / YOLO の高速化）

### インストール

```bash
git clone https://github.com/hirorogo/Movie-serch.git
cd Movie-serch
pip install -r requirements.txt
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

### 単一動画の解析

```bash
python src/main.py analyze --video path/to/video.mp4
```

### フォルダ一括解析

```bash
python src/main.py analyze --dir path/to/videos/
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

## 開発ロードマップ

### Phase 1: 基盤構築
- [x] README 作成
- [ ] プロジェクト構造・依存関係の定義
- [ ] 動画 → 音声抽出モジュール

### Phase 2: 声紋分析（主軸）
- [ ] 基準音声の登録・声紋ベクトル生成
- [ ] 話者ダイアライゼーション（区間分離）
- [ ] 声紋照合による出演者判定
- [ ] 2人のみ出演のケースへの対応

### Phase 3: 視覚分析（補助）
- [ ] フレーム画像抽出
- [ ] 人物検出（YOLO）
- [ ] 体型・髪型・服装の特徴抽出
- [ ] 視覚的特徴による補助判定

### Phase 4: 統合・出力
- [ ] 声紋 + 視覚スコアの統合判定ロジック
- [ ] JSON / CSV レポート出力
- [ ] CLI インターフェース完成

### Phase 5: 精度向上
- [ ] 閾値チューニング
- [ ] テストデータでの検証
- [ ] エッジケース対応（BGM が大きい、複数人同時発話 等）

## 設定ファイル（config.yaml）

```yaml
performers:
  - id: person_a
    name: "出演者A"
  - id: person_b
    name: "出演者B"
  - id: person_c
    name: "出演者C"

thresholds:
  voice_similarity: 0.75       # 声紋一致と判定する最低スコア
  visual_similarity: 0.60      # 視覚一致と判定する最低スコア
  combined_weight_voice: 0.7   # 総合判定での声紋の重み
  combined_weight_visual: 0.3  # 総合判定での視覚の重み

diarization:
  min_segment_duration: 1.0    # 最短発話区間（秒）
  max_speakers: 3              # 最大話者数

output:
  format: "json"               # json / csv
  dir: "output/"
```

## ライセンス

このプロジェクトは個人利用を目的としています。
