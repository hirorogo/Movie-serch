"""CLI エントリーポイント - 動画出演者分析ツール"""

import logging
import sys
from pathlib import Path

import click

from src.pipeline import AnalysisPipeline
from src.output.reporter import save_results, print_summary


def _setup_logging(verbose: bool = False) -> None:
    """ロギングの初期化"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-V", is_flag=True, help="デバッグログを表示する")
def cli(verbose):
    """アリスホリック動画 出演者分析ツール"""
    _setup_logging(verbose=verbose)


@cli.command()
@click.option("--video", "-v", type=click.Path(exists=True),
              help="解析対象の動画ファイル")
@click.option("--dir", "-d", "video_dir", type=click.Path(exists=True),
              help="解析対象の動画フォルダ")
@click.option("--config", "-c", default="config.yaml",
              help="設定ファイルのパス")
@click.option("--output", "-o", default="output",
              help="出力ディレクトリ")
@click.option("--format", "-f", "fmt", default="json",
              type=click.Choice(["json", "csv", "both"]),
              help="出力形式")
@click.option("--visual/--no-visual", default=False,
              help="視覚分析を有効にする")
@click.option("--hf-token", envvar="HF_TOKEN", default=None,
              help="HuggingFace トークン（pyannote用）")
def analyze(video, video_dir, config, output, fmt, visual, hf_token):
    """動画を解析して出演者を判定する。"""
    from src.preflight import run_preflight, PreflightError

    if not video and not video_dir:
        click.echo("エラー: --video または --dir を指定してください。", err=True)
        sys.exit(1)

    try:
        run_preflight(check_gpu_available=visual)
    except PreflightError as e:
        click.echo(f"エラー: {e}", err=True)
        sys.exit(1)

    click.echo("パイプラインを初期化中...")
    pipeline = AnalysisPipeline(config_path=config)
    pipeline.setup(enable_visual=visual, hf_token=hf_token)

    results = []

    if video:
        click.echo(f"解析中: {video}")
        result = pipeline.analyze_video(video)
        results.append(result)
    elif video_dir:
        click.echo(f"フォルダ解析中: {video_dir}")
        results = pipeline.analyze_batch(video_dir)

    if not results:
        click.echo("解析対象の動画が見つかりませんでした。")
        return

    # コンソール出力
    print_summary(results)

    # ファイル出力
    saved = save_results(results, output, fmt=fmt)
    for path in saved:
        click.echo(f"結果を保存しました: {path}")


@cli.command()
@click.option("--dir", "-d", "ref_dir", default="data/reference_voices",
              help="基準音声ディレクトリ")
def list_speakers(ref_dir):
    """登録済みの基準音声を一覧表示する。"""
    ref_path = Path(ref_dir)
    if not ref_path.exists():
        click.echo(f"ディレクトリが見つかりません: {ref_path}")
        return

    click.echo(f"\n基準音声ディレクトリ: {ref_path}\n")
    for speaker_dir in sorted(ref_path.iterdir()):
        if not speaker_dir.is_dir():
            continue

        audio_files = list(speaker_dir.glob("*.wav")) + list(speaker_dir.glob("*.mp3"))
        status = f"{len(audio_files)} ファイル" if audio_files else "未登録"
        click.echo(f"  {speaker_dir.name}: {status}")
        for f in audio_files:
            click.echo(f"    - {f.name}")

    click.echo()


@cli.command()
@click.option("--video", "-v", type=click.Path(exists=True), required=True,
              help="出演者がわかっている動画ファイル")
@click.option("--config", "-c", default="config.yaml",
              help="設定ファイルのパス")
@click.option("--hf-token", envvar="HF_TOKEN", default=None,
              help="HuggingFace トークン（pyannote用）")
def setup(video, config, hf_token):
    """初回セットアップ: 動画から話者を自動分離し、ラベル付けで基準データを作成。"""
    from src.setup_wizard import SetupWizard

    wizard = SetupWizard(config_path=config)
    wizard.run(video, hf_token=hf_token)


@cli.command(name="auto-analyze")
@click.option("--dir", "-d", "video_dir", type=click.Path(exists=True), required=True,
              help="動画フォルダのパス（中の動画を全て自動解析）")
@click.option("--config", "-c", default="config.yaml",
              help="設定ファイルのパス")
@click.option("--output", "-o", default="output",
              help="出力ディレクトリ")
@click.option("--format", "-f", "fmt", default="both",
              type=click.Choice(["json", "csv", "both"]),
              help="出力形式")
@click.option("--visual/--no-visual", default=False,
              help="視覚分析を有効にする")
@click.option("--hf-token", envvar="HF_TOKEN", default=None,
              help="HuggingFace トークン（pyannote用）")
@click.option("--skip-analyzed/--no-skip", default=True,
              help="解析済みの動画をスキップする（デフォルト: スキップ）")
@click.option("--recursive/--no-recursive", default=False,
              help="サブフォルダも再帰的に検索する")
def auto_analyze(video_dir, config, output, fmt, visual, hf_token, skip_analyzed, recursive):
    """過去の動画を全て放り込んで自動解析する。

    指定フォルダ内の全動画を自動で解析し、結果を出力します。
    既に解析済みの動画はスキップされるため、新しい動画を追加して
    再実行するだけで差分解析が可能です。

    \b
    使い方:
      python src/main.py auto-analyze --dir /path/to/videos/
      python src/main.py auto-analyze --dir /path/to/videos/ --recursive
    """
    from src.preflight import run_preflight, PreflightError

    try:
        run_preflight(check_gpu_available=visual)
    except PreflightError as e:
        click.echo(f"エラー: {e}", err=True)
        sys.exit(1)

    # 動画ファイルを収集
    video_dir_path = Path(video_dir)
    extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}

    if recursive:
        videos = sorted(
            f for f in video_dir_path.rglob("*")
            if f.suffix.lower() in extensions and f.is_file()
        )
    else:
        videos = sorted(
            f for f in video_dir_path.iterdir()
            if f.suffix.lower() in extensions and f.is_file()
        )

    if not videos:
        click.echo(f"動画ファイルが見つかりません: {video_dir}")
        return

    click.echo(f"\n動画 {len(videos)} 件を検出しました。")

    # パイプライン初期化
    click.echo("パイプラインを初期化中...")
    pipeline = AnalysisPipeline(config_path=config)
    pipeline.setup(enable_visual=visual, hf_token=hf_token)

    # 解析済みの動画名を取得
    analyzed_names: set[str] = set()
    if skip_analyzed:
        analyzed_names = pipeline._load_analyzed_names(output)
        if analyzed_names:
            click.echo(f"解析済み: {len(analyzed_names)} 件（スキップ）")

    # 解析実行
    all_results = []
    new_count = 0
    for i, video in enumerate(videos, 1):
        if video.name in analyzed_names:
            click.echo(f"  [{i}/{len(videos)}] スキップ: {video.name}")
            continue

        click.echo(f"  [{i}/{len(videos)}] 解析中: {video.name}")
        result = pipeline.analyze_video(str(video))
        all_results.append(result)
        new_count += 1

        # 途中結果を逐次保存（中断に備える）
        _save_incremental(all_results, output, fmt, analyzed_names)

    if not all_results:
        click.echo("\n新しく解析する動画はありません。全て解析済みです。")
        return

    # 最終サマリー
    print_summary(all_results)
    click.echo(f"\n新規解析: {new_count} 件 / 合計: {len(videos)} 件")

    saved = save_results(all_results, output, fmt=fmt)
    for path in saved:
        click.echo(f"結果を保存しました: {path}")


def _save_incremental(results, output_dir, fmt, existing_names):
    """中間結果を保存する（中断復帰用）。"""
    try:
        save_results(results, output_dir, fmt=fmt)
    except Exception:
        pass  # 途中保存の失敗は無視


@cli.command()
@click.option("--video", "-v", type=click.Path(exists=True), required=True,
              help="テスト対象の動画ファイル")
@click.option("--config", "-c", default="config.yaml",
              help="設定ファイルのパス")
def test_voice(video, config):
    """声紋照合のテスト実行（ダイアライゼーションなし）。"""
    from src.audio.extractor import extract_audio
    from src.audio.voice_matcher import VoiceMatcher

    import yaml
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    click.echo("声紋モデルを読み込み中...")
    matcher = VoiceMatcher(threshold=cfg["thresholds"]["voice_similarity"])
    matcher.register_speakers_from_dir(cfg["paths"]["reference_voices"])

    if not matcher.reference_embeddings:
        click.echo("エラー: 基準音声が登録されていません。data/reference_voices/ に音声ファイルを配置してください。")
        return

    click.echo(f"登録話者: {list(matcher.reference_embeddings.keys())}")

    click.echo(f"\n音声を抽出中: {video}")
    audio_path = extract_audio(video, sample_rate=cfg["audio"]["sample_rate"])

    click.echo("声紋照合中...")
    scores = matcher.compare(str(audio_path))

    click.echo(f"\n声紋類似度スコア:")
    threshold = cfg["thresholds"]["voice_similarity"]
    for sid, score in sorted(scores.items(), key=lambda x: -x[1]):
        mark = "○" if score >= threshold else "×"
        click.echo(f"  {mark} {sid}: {score:.4f}")

    # クリーンアップ
    Path(audio_path).unlink(missing_ok=True)


if __name__ == "__main__":
    cli()
