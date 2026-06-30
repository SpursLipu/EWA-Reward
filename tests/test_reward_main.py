import pytest
from fastapi.testclient import TestClient

import reward_main


def test_sample_uniform_frames_handles_empty_input():
    """中文：验证空帧序列采样会返回空列表。
English: Verify that sampling an empty frame sequence returns an empty list."""
    assert reward_main.sample_uniform_frames([], 4) == []


def test_sample_uniform_frames_keeps_first_and_last():
    """中文：验证均匀采样会保留首帧和尾帧。
English: Verify that uniform sampling preserves the first and last frames."""
    frames = ["f0", "f1", "f2", "f3", "f4"]

    sampled = reward_main.sample_uniform_frames(frames, 3)

    assert sampled[0] == "f0"
    assert sampled[-1] == "f4"
    assert len(sampled) == 3


def test_health_endpoint():
    """中文：验证健康检查接口返回服务状态。
English: Verify that the health endpoint returns service status."""
    client = TestClient(reward_main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "api_base" in response.json()


@pytest.mark.parametrize(
    "payload",
    [
        {"video_path": [], "prompt": "task"},
        {"video_path": ["/tmp/not-exist.mp4"], "prompt": "task"},
        {"video_path": ["/tmp/not-exist.mp4"], "prompt": ""},
    ],
)
def test_eval_video_rejects_bad_requests(payload):
    """中文：验证评估接口会拒绝非法请求。
English: Verify that the evaluation endpoint rejects invalid requests."""
    client = TestClient(reward_main.app)

    response = client.post("/eval_video", json=payload)

    assert response.status_code == 400
