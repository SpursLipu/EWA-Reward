"""Input and result persistence helpers."""

from __future__ import annotations

import os
import shutil
import textwrap
import threading
import uuid
from datetime import datetime, timedelta, timezone


file_lock = threading.Lock()
BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


async def save_video_and_prompt(video_list: list[str], shared_prompt: str, log_root: str):
    """中文：为一次评估创建日志目录，并保存输入 prompt 与视频副本。
English: Create a run log directory and persist the input prompt and copied videos."""
    os.makedirs(log_root, exist_ok=True)

    beijing_time = datetime.now(timezone.utc).astimezone(BEIJING_TIMEZONE)
    current_time = beijing_time.strftime("Y-%m-%d_%H:%M:%S")
    folder_id = current_time + "_" + str(uuid.uuid4())[:8]
    current_folder = os.path.join(log_root, folder_id)
    os.makedirs(current_folder, exist_ok=True)

    prompt_path = os.path.join(current_folder, "prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(shared_prompt)

    for idx, video_path in enumerate(video_list):
        video_filename = f"video_{idx}.mp4"
        video_save_path = os.path.join(current_folder, video_filename)
        shutil.copy2(video_path, video_save_path)

    return current_folder


def save_result_to_txt_table(current_folder, res):
    """中文：将各 reward 模块的 JSON 与接口结果写入可读文本报告。
English: Write module-level reward JSON and API outputs into a readable text report."""
    prompt_path = os.path.join(current_folder, "result_qwen36_27b.txt")
    max_width = 100

    def wrap_text(text):
        """中文：按报告宽度对长文本换行并缩进。
English: Wrap and indent long text for the report width."""
        lines = textwrap.wrap(str(text), width=max_width - 4)
        return "\n".join(["    " + line for line in lines])

    def write_module(f, title, json_result, api_result):
        """中文：写入单个 reward 模块的报告段落。
English: Write one reward module section into the report."""
        f.write("\n" + "*" * max_width + "\n")
        f.write(title.center(max_width) + "\n")
        f.write("*" * max_width + "\n")
        f.write("JSON提取结果:\n")
        f.write(wrap_text(json_result) + "\n")
        f.write("接口返回结果:\n")
        f.write(wrap_text(api_result) + "\n")
        f.write("*" * max_width + "\n")

    with file_lock:
        with open(prompt_path, "a", encoding="utf-8") as f:
            write_module(f, "Planning", res["planning_json_output"], res["planning_api_output"])

            write_module(
                f,
                "视频清晰度和亮度",
                res["clarity_and_brightness_json_output"],
                res["clarity_and_brightness_api_output"],
            )
            write_module(
                f,
                "视频清晰度和亮度reflection",
                res["clarity_and_brightness_reflection_json_output"],
                res["clarity_and_brightness_reflection_api_output"],
            )
            write_module(f, "视频色彩", res["color_json_output"], res["color_api_output"])
            write_module(f, "首帧一致", res["first_frame_json_output"], res["first_frame_api_output"])
            write_module(
                f,
                "异常帧检测",
                res["abnormal_frame_json_output"],
                res["abnormal_frame_api_output"],
            )

            write_module(
                f,
                "动作质量",
                res["motion_quality_score"],
                res["motion_quality_score"],
            )

            write_module(f, "交互一致性", res["interaction_json_output"], res["interaction_output"])
            write_module(f, "穿模", res["interpenetration_json_output"], res["interpenetration_output"])
            write_module(f, "形变", res["shape_json_output"], res["shape_output"])

            write_module(
                f,
                "指令跟随",
                res["instruction_following_json_output"],
                res["instruction_following_api_output"],
            )
            write_module(
                f,
                "指令跟随reflection",
                res["instruction_reflection_following_json_output"],
                res["instruction_reflection_following_api_output"],
            )
            write_module(
                f,
                "背景一致性",
                res["background_consistency_json_output"],
                res["background_consistency_api_output"],
            )
            write_module(f, "任务完成度", res["task_json_output"], res["task_api_output"])

            f.write("\n" + "*" * max_width + "\n")
            f.write("Total Score".center(max_width) + "\n")
            f.write("*" * max_width + "\n")
            f.write(str(res["total_score"]).center(max_width) + "\n\n")
            f.write("\n" * 4)
