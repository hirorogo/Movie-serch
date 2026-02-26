"""セットアップウィザード - 動画から自動で話者を分離し、AIが自動ラベリング → ユーザーは確認するだけ"""

import shutil
from pathlib import Path

import yaml
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

from src.audio.extractor import extract_audio, extract_audio_segment, get_video_duration
from src.audio.diarizer import Diarizer, SpeakerSegment


class SetupWizard:
    """初回セットアップを半自動で行うウィザード"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        self.encoder = VoiceEncoder()

    def run(self, video_path: str, hf_token: str | None = None) -> None:
        """セットアップウィザードを実行する。

        Args:
            video_path: 出演者がわかっている動画のパス
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画が見つかりません: {video_path}")

        print(f"\n{'='*60}")
        print(" セットアップウィザード")
        print(f"{'='*60}")
        print(f"\n対象動画: {video_path.name}")

        # Step 1: 音声抽出
        print("\n[Step 1/6] 音声を抽出中...")
        audio_path = extract_audio(
            str(video_path),
            sample_rate=self.config["audio"]["sample_rate"],
        )
        duration = get_video_duration(str(video_path))
        print(f"  動画の長さ: {int(duration//60)}分{int(duration%60)}秒")

        # Step 2: 話者ダイアライゼーション
        print("\n[Step 2/6] 話者を自動分離中...")
        diarizer = Diarizer(
            max_speakers=self.config["diarization"]["max_speakers"],
            min_segment_duration=self.config["diarization"]["min_segment_duration"],
            hf_token=hf_token,
        )
        segments = diarizer.diarize(str(audio_path))

        if not segments:
            print("  エラー: 話者を検出できませんでした。")
            Path(audio_path).unlink(missing_ok=True)
            return

        # クラスタごとに整理
        clusters = self._group_segments(segments)
        print(f"  {len(clusters)} 人の話者を検出しました。")

        # Step 3: 各クラスタの代表音声を保存 & 声紋ベクトル生成
        print("\n[Step 3/6] 各話者の音声サンプルを作成中...")
        sample_dir = Path("data/setup_samples")
        sample_dir.mkdir(parents=True, exist_ok=True)
        cluster_samples = self._save_cluster_samples(
            str(video_path), clusters, str(sample_dir)
        )

        # Step 4: 声の特徴を分析
        print("\n[Step 4/6] 各話者の声の特徴を分析中...")
        cluster_profiles = self._analyze_voice_profiles(cluster_samples)

        # Step 5: AI自動ラベリング or 手動ラベリング
        performer_ids = [p["id"] for p in self.config["performers"]]
        performer_names = {p["id"]: p["name"] for p in self.config["performers"]}

        # 既存の基準音声があるか確認
        existing_refs = self._load_existing_references()

        if existing_refs:
            print(f"\n[Step 5/6] AIが既存の基準音声と照合して自動判定中...")
            assignments = self._auto_label_with_references(
                cluster_samples, cluster_profiles, existing_refs,
                performer_ids, performer_names
            )
        else:
            print(f"\n[Step 5/6] AIが声の特徴から自動判定中...")
            assignments = self._auto_label_by_features(
                cluster_samples, cluster_profiles,
                performer_ids, performer_names
            )

        if not assignments:
            print("\n  ラベル付けがキャンセルされました。")
            self._cleanup(sample_dir, audio_path)
            return

        # Step 6: 基準音声として保存
        print(f"\n[Step 6/6] 基準音声を保存中...")
        self._save_references(str(video_path), clusters, assignments)

        # クリーンアップ
        self._cleanup(sample_dir, audio_path)

        print(f"\n{'='*60}")
        print(" セットアップ完了!")
        print(f"{'='*60}")
        print(f"\n  保存先: {self.config['paths']['reference_voices']}/")
        for cluster_label, person_id in assignments.items():
            name = performer_names.get(person_id, person_id)
            print(f"    {person_id} ({name}) ← 話者クラスタ {cluster_label}")
        print(f"\n  次のコマンドで解析を実行できます:")
        print(f"    python src/main.py analyze --video <動画ファイル>")
        print()

    # =========================================================================
    # 声の特徴分析
    # =========================================================================

    def _analyze_voice_profiles(self, cluster_samples: dict[str, dict]) -> dict[str, dict]:
        """各クラスタの声の特徴プロファイルを生成する。

        Returns:
            {クラスタラベル: {
                "embedding": np.ndarray,   # 声紋ベクトル（平均）
                "pitch_est": str,          # 声の高さ推定（"高め", "中間", "低め"）
                "total_time": float,       # 合計発話時間
                "segment_count": int,      # 発話区間数
            }}
        """
        import librosa

        profiles = {}

        for label, info in sorted(cluster_samples.items()):
            embeddings = []
            pitches = []

            for sp in info["sample_paths"]:
                # 声紋ベクトル
                wav = preprocess_wav(Path(sp))
                if len(wav) > 0:
                    emb = self.encoder.embed_utterance(wav)
                    embeddings.append(emb)

                # ピッチ推定
                try:
                    y, sr = librosa.load(sp, sr=16000)
                    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
                    valid_f0 = f0[~np.isnan(f0)]
                    if len(valid_f0) > 0:
                        pitches.append(float(np.median(valid_f0)))
                except Exception:
                    pass

            avg_embedding = np.mean(embeddings, axis=0) if embeddings else np.zeros(256)
            avg_pitch = float(np.mean(pitches)) if pitches else 0.0

            # ピッチを分類
            if avg_pitch > 200:
                pitch_label = "高め"
            elif avg_pitch > 140:
                pitch_label = "中間"
            elif avg_pitch > 0:
                pitch_label = "低め"
            else:
                pitch_label = "不明"

            profiles[label] = {
                "embedding": avg_embedding,
                "pitch_hz": avg_pitch,
                "pitch_est": pitch_label,
                "total_time": info["total_time"],
                "segment_count": info["segment_count"],
            }

            print(f"  {label}: 声の高さ={pitch_label} ({avg_pitch:.0f}Hz), "
                  f"発話量={info['total_time']:.1f}秒 ({info['segment_count']}区間)")

        return profiles

    def _load_existing_references(self) -> dict[str, np.ndarray]:
        """既存の基準音声があれば声紋ベクトルを読み込む。

        Returns:
            {person_id: 声紋ベクトル} の辞書。なければ空辞書。
        """
        ref_dir = Path(self.config["paths"]["reference_voices"])
        refs = {}

        if not ref_dir.exists():
            return refs

        for person_dir in sorted(ref_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            audio_files = list(person_dir.glob("*.wav")) + list(person_dir.glob("*.mp3"))
            if not audio_files:
                continue

            embeddings = []
            for af in audio_files:
                wav = preprocess_wav(af)
                if len(wav) > 0:
                    emb = self.encoder.embed_utterance(wav)
                    embeddings.append(emb)

            if embeddings:
                refs[person_dir.name] = np.mean(embeddings, axis=0)

        return refs

    # =========================================================================
    # AI自動ラベリング（既存基準あり）
    # =========================================================================

    def _auto_label_with_references(self, cluster_samples: dict[str, dict],
                                    cluster_profiles: dict[str, dict],
                                    existing_refs: dict[str, np.ndarray],
                                    performer_ids: list[str],
                                    performer_names: dict[str, str]) -> dict[str, str]:
        """既存の基準音声と照合してAIが自動でラベルを割り当てる。

        ユーザーには結果を確認してもらうだけ。
        """
        # 各クラスタと各基準音声の類似度を計算
        similarity_matrix = {}
        for label, profile in cluster_profiles.items():
            similarity_matrix[label] = {}
            for pid, ref_emb in existing_refs.items():
                emb = profile["embedding"]
                sim = float(np.dot(emb, ref_emb) / (
                    np.linalg.norm(emb) * np.linalg.norm(ref_emb) + 1e-10
                ))
                similarity_matrix[label][pid] = sim

        # 最適な割り当てを計算（ハンガリアン法的に貪欲マッチング）
        assignments = self._greedy_match(similarity_matrix)

        # 結果を表示してユーザーに確認
        return self._confirm_assignments(
            assignments, similarity_matrix, cluster_profiles,
            performer_names, mode="reference"
        )

    # =========================================================================
    # AI自動ラベリング（初回・基準なし）
    # =========================================================================

    def _auto_label_by_features(self, cluster_samples: dict[str, dict],
                                cluster_profiles: dict[str, dict],
                                performer_ids: list[str],
                                performer_names: dict[str, str]) -> dict[str, str]:
        """基準音声がない場合、声の特徴（ピッチ・発話量）でAIが自動で
        クラスタを区別し、仮ラベルを割り当てる。ユーザーに確認してもらう。
        """
        # 声の特徴でソート（ピッチが高い順）してマッピング
        sorted_clusters = sorted(
            cluster_profiles.items(),
            key=lambda x: (-x[1]["pitch_hz"], -x[1]["total_time"])
        )

        # クラスタ数と出演者数の少ない方に合わせる
        n = min(len(sorted_clusters), len(performer_ids))
        assignments = {}
        for i in range(n):
            label = sorted_clusters[i][0]
            pid = performer_ids[i]
            assignments[label] = pid

        # 類似度マトリクスはないので特徴情報のみで確認
        return self._confirm_assignments_initial(
            assignments, cluster_profiles, performer_names
        )

    # =========================================================================
    # ユーザー確認
    # =========================================================================

    def _confirm_assignments(self, assignments: dict[str, str],
                             similarity_matrix: dict[str, dict[str, float]],
                             cluster_profiles: dict[str, dict],
                             performer_names: dict[str, str],
                             mode: str = "reference") -> dict[str, str]:
        """AIの自動ラベリング結果をユーザーに確認してもらう。"""

        print(f"\n  {'─'*50}")
        print(f"  AIの自動判定結果:")
        print(f"  {'─'*50}")

        for label, pid in sorted(assignments.items()):
            name = performer_names.get(pid, pid)
            profile = cluster_profiles.get(label, {})
            pitch = profile.get("pitch_est", "?")
            total_time = profile.get("total_time", 0)

            sim = similarity_matrix.get(label, {}).get(pid, 0.0)
            confidence = "高" if sim >= 0.85 else "中" if sim >= 0.70 else "低"

            print(f"    {label} → {pid} ({name})")
            print(f"      類似度: {sim:.4f} [{confidence}]  "
                  f"声の高さ: {pitch}  発話量: {total_time:.1f}秒")

        # 割り当てられなかったクラスタがあれば表示
        unassigned = set(cluster_profiles.keys()) - set(assignments.keys())
        if unassigned:
            print(f"\n    未割当クラスタ: {', '.join(sorted(unassigned))} (この動画に出ていない人)")

        return self._ask_confirmation(assignments, cluster_profiles, performer_names)

    def _confirm_assignments_initial(self, assignments: dict[str, str],
                                     cluster_profiles: dict[str, dict],
                                     performer_names: dict[str, str]) -> dict[str, str]:
        """初回セットアップ時のAI判定結果をユーザーに確認してもらう。"""

        print(f"\n  {'─'*50}")
        print(f"  AIの自動判定結果（声の特徴に基づく仮割当）:")
        print(f"  {'─'*50}")
        print(f"  ※ 初回のため声の高さ・発話量で仮の順番で割り当てています")

        for label, pid in sorted(assignments.items()):
            name = performer_names.get(pid, pid)
            profile = cluster_profiles.get(label, {})
            pitch = profile.get("pitch_est", "?")
            pitch_hz = profile.get("pitch_hz", 0)
            total_time = profile.get("total_time", 0)

            print(f"    {label} → {pid} ({name})")
            print(f"      声の高さ: {pitch} ({pitch_hz:.0f}Hz)  "
                  f"発話量: {total_time:.1f}秒")

        unassigned = set(cluster_profiles.keys()) - set(assignments.keys())
        if unassigned:
            print(f"\n    未割当クラスタ: {', '.join(sorted(unassigned))} (この動画に出ていない人)")

        return self._ask_confirmation(assignments, cluster_profiles, performer_names)

    def _ask_confirmation(self, assignments: dict[str, str],
                          cluster_profiles: dict[str, dict],
                          performer_names: dict[str, str]) -> dict[str, str]:
        """確認・修正の対話ループ。

        Returns:
            最終的な割り当て辞書。キャンセル時は空辞書。
        """
        while True:
            print(f"\n  操作を選択してください:")
            print(f"    y  → この割り当てで決定")
            print(f"    e  → 手動で修正する")
            print(f"    q  → キャンセル")

            try:
                choice = input(f"\n  [y/e/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return {}

            if choice == "y":
                print(f"  → 割り当てを確定しました。")
                return assignments

            elif choice == "q":
                return {}

            elif choice == "e":
                assignments = self._manual_edit(
                    assignments, cluster_profiles, performer_names
                )
                if not assignments:
                    return {}

                # 修正後の結果を再表示
                print(f"\n  修正後の割り当て:")
                for label, pid in sorted(assignments.items()):
                    name = performer_names.get(pid, pid)
                    print(f"    {label} → {pid} ({name})")
                # ループに戻って再度確認
            else:
                print(f"  y, e, q のいずれかを入力してください。")

    def _manual_edit(self, assignments: dict[str, str],
                     cluster_profiles: dict[str, dict],
                     performer_names: dict[str, str]) -> dict[str, str]:
        """手動修正モード: 特定のクラスタの割り当てを変更する。"""

        performer_ids = list(performer_names.keys())
        all_labels = sorted(cluster_profiles.keys())

        print(f"\n  修正するクラスタを選択してください:")
        for i, label in enumerate(all_labels):
            current = assignments.get(label, "(未割当)")
            if current != "(未割当)":
                current = f"{current} ({performer_names.get(current, current)})"
            print(f"    {i+1}. {label} → 現在: {current}")
        print(f"    d  → 完了")

        new_assignments = dict(assignments)

        while True:
            try:
                choice = input(f"\n  修正するクラスタ番号 [1-{len(all_labels)}/d]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return {}

            if choice == "d":
                return new_assignments

            try:
                idx = int(choice) - 1
                if not (0 <= idx < len(all_labels)):
                    print(f"  1〜{len(all_labels)} の数字を入力してください。")
                    continue
            except ValueError:
                print(f"  数字または 'd' を入力してください。")
                continue

            target_label = all_labels[idx]

            # 割り当て先を選択
            print(f"\n  {target_label} をどの出演者に割り当てますか？")
            for i, pid in enumerate(performer_ids):
                name = performer_names.get(pid, pid)
                print(f"    {i+1}. {pid} ({name})")
            print(f"    s. 未割当にする")

            while True:
                try:
                    pid_choice = input(f"  選択 [1-{len(performer_ids)}/s]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return {}

                if pid_choice == "s":
                    new_assignments.pop(target_label, None)
                    print(f"  → {target_label} を未割当にしました。")
                    break

                try:
                    pid_idx = int(pid_choice) - 1
                    if 0 <= pid_idx < len(performer_ids):
                        new_pid = performer_ids[pid_idx]
                        # 既に他のクラスタに割り当てられていたら外す
                        for lbl, pid in list(new_assignments.items()):
                            if pid == new_pid and lbl != target_label:
                                del new_assignments[lbl]
                                print(f"  ({lbl} の割り当てを解除しました)")
                        new_assignments[target_label] = new_pid
                        name = performer_names.get(new_pid, new_pid)
                        print(f"  → {target_label} = {new_pid} ({name})")
                        break
                    else:
                        print(f"  1〜{len(performer_ids)} の数字を入力してください。")
                except ValueError:
                    print(f"  数字または 's' を入力してください。")

        return new_assignments

    # =========================================================================
    # ユーティリティ
    # =========================================================================

    def _greedy_match(self, similarity_matrix: dict[str, dict[str, float]]) -> dict[str, str]:
        """類似度マトリクスから貪欲法で最適な割り当てを計算する。"""
        assignments = {}
        used_pids = set()

        # 全ペアをスコア順でソート
        pairs = []
        for label, scores in similarity_matrix.items():
            for pid, sim in scores.items():
                pairs.append((sim, label, pid))

        pairs.sort(reverse=True)

        for sim, label, pid in pairs:
            if label in assignments or pid in used_pids:
                continue
            assignments[label] = pid
            used_pids.add(pid)

        return assignments

    def _group_segments(self, segments: list[SpeakerSegment]) -> dict[str, list[SpeakerSegment]]:
        """セグメントを話者ラベルごとにグループ化"""
        clusters: dict[str, list[SpeakerSegment]] = {}
        for seg in segments:
            if seg.speaker_label not in clusters:
                clusters[seg.speaker_label] = []
            clusters[seg.speaker_label].append(seg)
        return clusters

    def _save_cluster_samples(self, video_path: str,
                              clusters: dict[str, list[SpeakerSegment]],
                              output_dir: str) -> dict[str, dict]:
        """各クラスタの代表音声サンプルを保存する。

        Returns:
            {クラスタラベル: {"sample_paths": [...], "total_time": float, "segment_count": int}}
        """
        result = {}

        for label, segs in sorted(clusters.items()):
            sorted_segs = sorted(segs, key=lambda s: s.duration, reverse=True)
            selected = sorted_segs[:3]

            sample_paths = []
            for i, seg in enumerate(selected):
                sample_path = Path(output_dir) / f"{label}_sample_{i}.wav"
                try:
                    extract_audio_segment(
                        video_path, seg.start, seg.end,
                        output_path=str(sample_path),
                        sample_rate=self.config["audio"]["sample_rate"],
                    )
                    sample_paths.append(str(sample_path))
                except Exception as e:
                    print(f"  警告: セグメント抽出失敗 ({label}, {seg.start:.1f}-{seg.end:.1f}s): {e}")

            total_time = sum(s.duration for s in segs)
            result[label] = {
                "sample_paths": sample_paths,
                "total_time": total_time,
                "segment_count": len(segs),
            }

            print(f"  話者 {label}: {len(segs)}区間, 合計 {total_time:.1f}秒")

        return result

    def _save_references(self, video_path: str,
                         clusters: dict[str, list[SpeakerSegment]],
                         assignments: dict[str, str]) -> None:
        """ラベル付けに基づいて基準音声を保存する。"""
        ref_dir = Path(self.config["paths"]["reference_voices"])

        for cluster_label, person_id in assignments.items():
            person_dir = ref_dir / person_id
            person_dir.mkdir(parents=True, exist_ok=True)

            segs = clusters[cluster_label]
            sorted_segs = sorted(segs, key=lambda s: s.duration, reverse=True)
            selected = sorted_segs[:5]

            for i, seg in enumerate(selected):
                output_path = person_dir / f"reference_{i}.wav"
                try:
                    extract_audio_segment(
                        video_path, seg.start, seg.end,
                        output_path=str(output_path),
                        sample_rate=self.config["audio"]["sample_rate"],
                    )
                    print(f"  保存: {output_path} ({seg.duration:.1f}秒)")
                except Exception as e:
                    print(f"  警告: 保存失敗 {output_path}: {e}")

    def _cleanup(self, sample_dir: Path, audio_path) -> None:
        """一時ファイルを削除"""
        try:
            shutil.rmtree(sample_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            Path(audio_path).unlink(missing_ok=True)
        except Exception:
            pass


def run_auto_setup(video_path: str, config_path: str = "config.yaml",
                   hf_token: str | None = None) -> None:
    """セットアップウィザードを実行するヘルパー関数"""
    wizard = SetupWizard(config_path=config_path)
    wizard.run(video_path, hf_token=hf_token)
