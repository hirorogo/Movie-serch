"""Web ダッシュボードのテスト"""

import json
import os
import tempfile

import pytest
import yaml

from src.web.app import create_app


SAMPLE_CONFIG = {
    "performers": [
        {"id": "person_a", "name": "A"},
        {"id": "person_b", "name": "B"},
    ],
    "thresholds": {
        "voice_similarity": 0.75,
        "visual_similarity": 0.60,
        "combined_weight_voice": 0.7,
        "combined_weight_visual": 0.3,
    },
    "paths": {
        "reference_voices": "data/reference_voices",
        "reference_visuals": "data/reference_visuals",
    },
}

SAMPLE_RESULTS = {
    "total_videos": 1,
    "results": [
        {
            "video": "test.mp4",
            "duration": "2:00",
            "performers": {
                "person_a": {
                    "name": "A",
                    "detected": True,
                    "voice_score": 0.88,
                    "visual_score": 0.65,
                    "combined_score": 0.81,
                    "speaking_time": "1:00",
                    "matching_segments": 3,
                },
                "person_b": {
                    "name": "B",
                    "detected": False,
                    "voice_score": 0.30,
                    "visual_score": 0.20,
                    "combined_score": 0.27,
                    "speaking_time": "0:00",
                    "matching_segments": 0,
                },
            },
            "detected_count": 1,
            "summary": "A",
            "errors": [],
        }
    ],
}


@pytest.fixture
def app_with_data():
    """テストデータ付きのFlaskアプリを生成"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG, f, allow_unicode=True)

        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_RESULTS, f)

        app = create_app(config_path=config_path, output_dir=output_dir)
        app.config["TESTING"] = True
        yield app


@pytest.fixture
def client(app_with_data):
    return app_with_data.test_client()


class TestPages:
    def test_dashboard(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Dashboard" in resp.data

    def test_results_page(self, client):
        resp = client.get("/results")
        assert resp.status_code == 200
        assert b"Results" in resp.data

    def test_stats_page(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert b"Stats" in resp.data

    def test_optimizer_page(self, client):
        resp = client.get("/optimizer")
        assert resp.status_code == 200
        assert b"Optimizer" in resp.data


class TestAPI:
    def test_overview(self, client):
        resp = client.get("/api/overview")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_videos"] == 1
        assert "thresholds" in data

    def test_results(self, client):
        resp = client.get("/api/results")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 1

    def test_result_detail(self, client):
        resp = client.get("/api/results/test.mp4")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["video"] == "test.mp4"

    def test_result_detail_not_found(self, client):
        resp = client.get("/api/results/nonexistent.mp4")
        assert resp.status_code == 404

    def test_performers(self, client):
        resp = client.get("/api/performers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "person_a" in data

    def test_matrix(self, client):
        resp = client.get("/api/matrix")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "performer_ids" in data
        assert "videos" in data

    def test_trends(self, client):
        resp = client.get("/api/trends")
        assert resp.status_code == 200

    def test_confidence(self, client):
        resp = client.get("/api/confidence")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "summary" in data

    def test_optimize(self, client):
        resp = client.get("/api/optimize")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "recommended_voice" in data

    def test_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "thresholds" in data

    def test_update_thresholds(self, client):
        resp = client.post(
            "/api/config/thresholds",
            json={"voice_similarity": 0.80, "visual_similarity": 0.65},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["thresholds"]["voice_similarity"] == 0.80

    def test_update_thresholds_no_data(self, client):
        resp = client.post(
            "/api/config/thresholds",
            json=None,
            content_type="application/json",
        )
        assert resp.status_code == 400
