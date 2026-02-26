"""分析パイプライン - 声紋分析と視覚分析を統合して出演者を判定"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.audio.extractor import extract_audio, extract_audio_segment, get_video_duration
from src.audio.voice_matcher import VoiceMatcher
from src.audio.diarizer import Diarizer


@dataclass
class PerformerResult:
    """出演者ごとの分析結果"""
    person_id: str
    name: str
    detected: bool
    voice_score: float = 0.0
    visual_score: float = 0.0
    combined_score: float = 0.0
    speaking_time: float = 0.0      # 発話時間（秒）
    matching_segments: int = 0


@dataclass
class VideoAnalysisResult:
    """動画の分析結果"""
    video_path: str
    video_name: str
    duration: float
    performers: list[PerformerResult] = field(default_factory=list)
    detected_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """辞書形式に変換"""
        performers_dict = {}
        for p in self.performers:
            performers_dict[p.person_id] = {
                "name": p.name,
                "detected": p.detected,
                "voice_score": round(p.voice_score, 4),
                "visual_score": round(p.visual_score, 4),
                "combined_score": round(p.combined_score, 4),
                "speaking_time": _format_time(p.speaking_time),
                "matching_segments": p.matching_segments,
            }

        detected_names = [p.name for p in self.performers if p.detected]
        summary = f"出演者: {', '.join(detected_names)}（{len(detected_names)}名）" if detected_names else "出演者なし"

        return {
            "video": self.video_name,
            "duration": _format_time(self.duration),
            "performers": performers_dict,
            "detected_count": self.detected_count,
            "summary": summary,
            "errors": self.errors,
        }


def _format_time(seconds: float) -> str:
    """秒数を m:ss 形式にフォーマット"""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


class AnalysisPipeline:
    """声紋 + 視覚分析を統合する解析パイプライン"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.voice_matcher = VoiceMatcher(
            threshold=self.config["thresholds"]["voice_similarity"]
        )
        self.diarizer = Diarizer(
            max_speakers=self.config["diarization"]["max_speakers"],
            min_segment_duration=self.config["diarization"]["min_segment_duration"],
        )

        # 視覚分析は任意（モデルが重いのでオプション）
        self.body_analyzer = None
        self.appearance_analyzer = None
        self.visual_enabled = False

        self.performers = self.config["performers"]

    def setup(self, enable_visual: bool = False, hf_token: str | None = None) -> None:
        """分析の初期化。基準データの読み込みとモデル準備。

        Args:
            enable_visual: 視覚分析を有効にするか
            hf_token: HuggingFace トークン（pyannote用）
        """
        # 声紋の基準データ登録
        ref_voices_dir = self.config["paths"]["reference_voices"]
        self.voice_matcher.register_speakers_from_dir(ref_voices_dir)

        # 視覚分析の初期化（オプション）
        if enable_visual:
            self._setup_visual()

    def _setup_visual(self) -> None:
        """視覚分析モジュールの初期化"""
        from src.visual.body_analyzer import BodyAnalyzer
        from src.visual.appearance import AppearanceAnalyzer

        self.body_analyzer = BodyAnalyzer(
            confidence_threshold=self.config["visual"]["confidence_threshold"]
        )
        self.appearance_analyzer = AppearanceAnalyzer(
            threshold=self.config["thresholds"]["visual_similarity"]
        )

        ref_visuals_dir = self.config["paths"]["reference_visuals"]
        if Path(ref_visuals_dir).exists():
            self.appearance_analyzer.register_references_from_dir(ref_visuals_dir)

        self.visual_enabled = True

    def analyze_video(self, video_path: str) -> VideoAnalysisResult:
        """単一動画を解析する。

        Args:
            video_path: 動画ファイルのパス

        Returns:
            VideoAnalysisResult
        """
        video_path_obj = Path(video_path)
        result = VideoAnalysisResult(
            video_path=str(video_path_obj),
            video_name=video_path_obj.name,
            duration=0.0,
        )

        try:
            result.duration = get_video_duration(video_path)
        except Exception as e:
            result.errors.append(f"動画情報取得エラー: {e}")
            return result

        # Step 1: 音声抽出
        try:
            audio_path = extract_audio(
                video_path,
                sample_rate=self.config["audio"]["sample_rate"],
            )
        except Exception as e:
            result.errors.append(f"音声抽出エラー: {e}")
            return result

        # Step 2: 話者ダイアライゼーション
        try:
            segments = self.diarizer.diarize(str(audio_path))
        except Exception as e:
            result.errors.append(f"ダイアライゼーションエラー: {e}")
            segments = []

        # Step 3: 声紋照合
        voice_results = self._analyze_voice(str(audio_path), video_path, segments)

        # Step 4: 視覚分析（有効な場合）
        visual_results = {}
        if self.visual_enabled:
            try:
                visual_results = self._analyze_visual(video_path)
            except Exception as e:
                result.errors.append(f"視覚分析エラー: {e}")

        # Step 5: 統合判定
        result.performers = self._combine_results(voice_results, visual_results)
        result.detected_count = sum(1 for p in result.performers if p.detected)

        # 一時ファイルのクリーンアップ
        try:
            Path(audio_path).unlink(missing_ok=True)
        except Exception:
            pass

        return result

    def _analyze_voice(self, audio_path: str, video_path: str,
                       segments: list) -> dict[str, dict]:
        """声紋分析を実行"""
        if not segments:
            # セグメントがない場合、全体を対象にする
            scores = self.voice_matcher.compare(audio_path)
            return {
                sid: {
                    "max_score": score,
                    "avg_score": score,
                    "matching_segments": 1 if score >= self.voice_matcher.threshold else 0,
                    "total_segments": 1,
                    "speaking_time": 0.0,
                }
                for sid, score in scores.items()
            }

        # 各セグメントの音声を抽出して照合
        segment_data = []
        for seg in segments:
            try:
                seg_audio = extract_audio_segment(
                    video_path, seg.start, seg.end,
                    sample_rate=self.config["audio"]["sample_rate"],
                )
                segment_data.append({
                    "start": seg.start,
                    "end": seg.end,
                    "audio_path": str(seg_audio),
                    "speaker_label": seg.speaker_label,
                })
            except Exception:
                continue

        if not segment_data:
            scores = self.voice_matcher.compare(audio_path)
            return {
                sid: {"max_score": score, "avg_score": score,
                      "matching_segments": 0, "total_segments": 0,
                      "speaking_time": 0.0}
                for sid, score in scores.items()
            }

        voice_results = self.voice_matcher.compare_segments(segment_data)

        # 各話者の発話時間を計算
        for sid in voice_results:
            speaking_time = 0.0
            for seg in segment_data:
                scores = self.voice_matcher.compare(seg["audio_path"])
                if scores.get(sid, 0) >= self.voice_matcher.threshold:
                    speaking_time += seg["end"] - seg["start"]
            voice_results[sid]["speaking_time"] = speaking_time

        # セグメント音声の一時ファイルをクリーンアップ
        for seg in segment_data:
            try:
                Path(seg["audio_path"]).unlink(missing_ok=True)
            except Exception:
                pass

        return voice_results

    def _analyze_visual(self, video_path: str) -> dict[str, dict]:
        """視覚分析を実行"""
        from src.visual.frame_extractor import extract_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            frames = extract_frames(
                video_path, tmpdir,
                interval_sec=self.config["visual"]["frame_interval"],
            )

            if not frames:
                return {}

            # 各フレームから人物を切り出し
            all_crops = []
            crops_dir = Path(tmpdir) / "crops"
            for frame in frames:
                crops = self.body_analyzer.extract_person_crops(
                    str(frame), str(crops_dir)
                )
                all_crops.extend(c["path"] for c in crops)

            if not all_crops or not self.appearance_analyzer.reference_features:
                return {}

            return self.appearance_analyzer.compare_crops(all_crops)

    def _combine_results(self, voice_results: dict[str, dict],
                         visual_results: dict[str, dict]) -> list[PerformerResult]:
        """声紋と視覚のスコアを統合して最終判定を行う"""
        weight_voice = self.config["thresholds"]["combined_weight_voice"]
        weight_visual = self.config["thresholds"]["combined_weight_visual"]
        voice_threshold = self.config["thresholds"]["voice_similarity"]

        results = []
        for performer in self.performers:
            pid = performer["id"]
            name = performer["name"]

            # 声紋スコア
            v_result = voice_results.get(pid, {})
            voice_score = v_result.get("max_score", 0.0)
            speaking_time = v_result.get("speaking_time", 0.0)
            matching_segments = v_result.get("matching_segments", 0)

            # 視覚スコア
            vis_result = visual_results.get(pid, {})
            visual_score = vis_result.get("max_score", 0.0)

            # 統合スコア
            if visual_results:
                combined = weight_voice * voice_score + weight_visual * visual_score
            else:
                combined = voice_score

            detected = combined >= voice_threshold

            results.append(PerformerResult(
                person_id=pid,
                name=name,
                detected=detected,
                voice_score=voice_score,
                visual_score=visual_score,
                combined_score=combined,
                speaking_time=speaking_time,
                matching_segments=matching_segments,
            ))

        return results

    def analyze_batch(self, video_dir: str) -> list[VideoAnalysisResult]:
        """フォルダ内の全動画を一括解析する。

        Args:
            video_dir: 動画フォルダのパス

        Returns:
            VideoAnalysisResult のリスト
        """
        video_dir = Path(video_dir)
        extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}

        videos = [
            f for f in sorted(video_dir.iterdir())
            if f.suffix.lower() in extensions
        ]

        results = []
        for video in videos:
            result = self.analyze_video(str(video))
            results.append(result)

        return results
