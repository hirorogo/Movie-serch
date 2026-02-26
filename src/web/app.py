"""Flask Web アプリケーション - 解析結果のダッシュボードと統計表示"""

import json
import logging
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template, request

from src.optimizer import ThresholdOptimizer
from src.stats import ResultsAnalyzer

logger = logging.getLogger(__name__)


def create_app(config_path: str = "config.yaml", output_dir: str = "output") -> Flask:
    """Flask アプリケーションファクトリ

    Args:
        config_path: config.yaml のパス
        output_dir: 解析結果の出力ディレクトリ
    """
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["CONFIG_PATH"] = config_path
    app.config["OUTPUT_DIR"] = output_dir

    def _load_config() -> dict:
        p = Path(config_path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_results() -> list[dict]:
        results_path = Path(output_dir) / "results.json"
        if not results_path.exists():
            return []
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("results", []) if isinstance(data, dict) else data

    # --- ページルート ---

    @app.route("/")
    def dashboard():
        """ダッシュボード（トップページ）"""
        return render_template("dashboard.html")

    @app.route("/results")
    def results_page():
        """解析結果一覧ページ"""
        return render_template("results.html")

    @app.route("/stats")
    def stats_page():
        """統計・分析ページ"""
        return render_template("stats.html")

    @app.route("/optimizer")
    def optimizer_page():
        """閾値最適化ページ"""
        return render_template("optimizer.html")

    # --- API エンドポイント ---

    @app.route("/api/overview")
    def api_overview():
        """全体概要API"""
        results = _load_results()
        analyzer = ResultsAnalyzer(results_data=results)
        config = _load_config()
        overview = analyzer.get_overview()
        overview["thresholds"] = config.get("thresholds", {})
        return jsonify(overview)

    @app.route("/api/results")
    def api_results():
        """解析結果一覧API"""
        results = _load_results()
        return jsonify({"results": results, "total": len(results)})

    @app.route("/api/results/<path:video_name>")
    def api_result_detail(video_name):
        """個別動画の結果API"""
        results = _load_results()
        analyzer = ResultsAnalyzer(results_data=results)
        detail = analyzer.get_video_details(video_name)
        if not detail:
            return jsonify({"error": "動画が見つかりません"}), 404
        return jsonify(detail[0])

    @app.route("/api/performers")
    def api_performers():
        """出演者分析API"""
        results = _load_results()
        analyzer = ResultsAnalyzer(results_data=results)
        return jsonify(analyzer.get_performer_analysis())

    @app.route("/api/matrix")
    def api_matrix():
        """出演マトリクスAPI"""
        results = _load_results()
        analyzer = ResultsAnalyzer(results_data=results)
        return jsonify(analyzer.get_detection_matrix())

    @app.route("/api/trends")
    def api_trends():
        """スコア推移API"""
        results = _load_results()
        analyzer = ResultsAnalyzer(results_data=results)
        return jsonify(analyzer.get_score_trends())

    @app.route("/api/confidence")
    def api_confidence():
        """信頼度分析API"""
        results = _load_results()
        analyzer = ResultsAnalyzer(results_data=results)
        return jsonify(analyzer.get_confidence_analysis())

    @app.route("/api/optimize")
    def api_optimize():
        """閾値最適化API"""
        results = _load_results()
        if not results:
            return jsonify({"error": "解析結果がありません。先に動画を解析してください。"}), 400
        optimizer = ThresholdOptimizer(results)
        return jsonify(optimizer.get_recommendation())

    @app.route("/api/config")
    def api_config():
        """現在の設定API"""
        return jsonify(_load_config())

    @app.route("/api/config/thresholds", methods=["POST"])
    def api_update_thresholds():
        """閾値設定の更新API"""
        data = request.get_json()
        if not data:
            return jsonify({"error": "リクエストデータがありません"}), 400

        config = _load_config()
        thresholds = config.get("thresholds", {})

        if "voice_similarity" in data:
            thresholds["voice_similarity"] = float(data["voice_similarity"])
        if "visual_similarity" in data:
            thresholds["visual_similarity"] = float(data["visual_similarity"])

        config["thresholds"] = thresholds

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        logger.info("閾値を更新: %s", thresholds)
        return jsonify({"status": "ok", "thresholds": thresholds})

    return app
