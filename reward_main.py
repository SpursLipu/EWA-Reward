import json
import uvicorn
import asyncio
import traceback
from pathlib import Path

from typing import List
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException

from prompt import *
from score_calc import *
from ewa_reward.config import get_settings
from ewa_reward.llm import retry_llm_call as retry_llm_call_base
from ewa_reward.logging import save_result_to_txt_table
from ewa_reward.logging import save_video_and_prompt as save_video_and_prompt_base
from ewa_reward.metrics import MotionQualityMetrics
from ewa_reward.video import (
    build_content,
    build_physics_reflection_content,
    extract_all_frames,
    sample_uniform_frames,
)

# os.environ["VLLM_DISABLE_FFMPEG=1"]

SETTINGS = get_settings()
BASE_ROOT = str(SETTINGS.log_root)
MOTION_QUALITY = MotionQualityMetrics(SETTINGS)
MOTION_QUALITY_AVAILABLE = False

app = FastAPI(
    title="EWA-Reward",
    description="Agentic reward evaluation service for embodied world-model videos.",
    version="0.1.0",
)

async def retry_llm_call(messages, idx, calc_score):
    """中文：使用全局配置执行带重试的模型调用与计分。
English: Run a retried model call and scoring step with global settings."""
    return await retry_llm_call_base(messages, idx, calc_score, SETTINGS)

async def planning(description, frames, idx):
    # 创建prompt
    """中文：评估视频是否具备继续执行完整 reward 流程的基础质量。
English: Judge whether the video has enough baseline quality for the full reward pipeline."""
    system_prompt = create_planning_prompt()
    content = build_content(frames, description)
    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, planning_calc_score)
    return json_output, api_output


async def instruction_following(description, frames, idx):
    # 创建prompt
    """中文：评估视频内容对用户任务指令的跟随程度。
English: Evaluate how well the video follows the user task instruction."""
    system_prompt = create_instruction_following_prompt()
    content = build_content(frames, description)
    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, instruction_following_calc_score)
    return json_output, api_output

async def background_consistency(description, frames, idx):
    # 创建prompt
    """中文：评估物体与环境背景是否与任务描述保持一致。
English: Evaluate whether objects and background remain consistent with the task description."""
    system_prompt = create_background_consistency_prompt()
    content = build_content(frames, description)
    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, background_consistency_calc_score)
    return json_output, api_output

async def instruction_following_reflection(description, frames, idx):
    # 创建prompt
    """中文：基于首次指令跟随结果进行反思校正。
English: Refine the initial instruction-following judgment through reflection."""
    system_prompt = create_instruction_following_prompt()
    reflection_prompt = create_instruction_following_reflection_prompt()
    
    description = json.dumps(description, ensure_ascii=False) 
    content = build_content(frames, description)
    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        },
        {
            "role": "assistant",
            "content": description
        },
        {
            "role": "user",
            "content": reflection_prompt
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, instruction_following_calc_score)

    return json_output, api_output

async def clarity_and_brightness(frames, idx):
    # 创建prompt
    """中文：评估视频帧的清晰度与亮度质量。
English: Evaluate frame clarity and brightness quality."""
    prompt = create_clarity_and_brightness_prompt()
    
    content = build_content(frames)
    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, clarity_and_brightness_calc_score)
    return json_output, api_output   

async def clarity_and_brightness_reflection(frames, idx):
    # 创建prompt
    """中文：对清晰度与亮度判断执行反思式复核。
English: Perform a reflection-style review for clarity and brightness judgments."""
    prompt = create_clarity_and_brightness_reflection_prompt()
    
    content = build_content(frames)
    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, clarity_and_brightness_calc_score)
    return json_output, api_output   

async def color(frames, idx):
    # 创建prompt
    """中文：评估视频色彩是否自然且稳定。
English: Evaluate whether video color is natural and stable."""
    color_system_prompt = create_color_quality_prompt()
    
    content = build_content(frames)
    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": color_system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, color_calc_score)
    return json_output, api_output

async def first_frame(frames, idx):
    # 创建prompt
    """中文：评估首帧是否与任务初始状态一致。
English: Evaluate whether the first frame matches the task's initial state."""
    system_prompt = create_first_frame_prompt()
    
    content = build_content(frames)
    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, color_calc_score)
    return json_output, api_output

async def abnormal_frame(frames, idx):
    # 创建prompt
    """中文：检测视频中是否存在明显异常帧。
English: Detect whether the video contains obvious abnormal frames."""
    system_prompt = create_abnormal_frame_prompt()
    
    content = build_content(frames)
    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, color_calc_score)
    return json_output, api_output

async def vision_quality(frames, idx):
    """中文：聚合视觉质量相关 reward 子模块的评估结果。
English: Aggregate reward submodules related to visual quality."""
    clarity_and_brightness_json_output, clarity_and_brightness_api_output = await clarity_and_brightness(frames, idx)
    clarity_and_brightness_reflection_json_output, clarity_and_brightness_reflection_api_output = None, None
    color_json_output, color_api_output = None, None
    first_frame_json_output, first_frame_api_output = None, None
    # abnormal_frame_json_output, abnormal_frame_api_output = await abnormal_frame(frames, idx)
    abnormal_frame_json_output, abnormal_frame_api_output = None, None

    return (clarity_and_brightness_json_output, clarity_and_brightness_api_output, clarity_and_brightness_reflection_json_output, clarity_and_brightness_reflection_api_output, color_json_output, color_api_output, first_frame_json_output, first_frame_api_output, abnormal_frame_json_output, abnormal_frame_api_output)

async def interaction(frames, idx):
    # 创建prompt
    """中文：评估物体交互和接触动态是否合理。
English: Evaluate whether object interactions and contact dynamics are plausible."""
    system_prompt = create_interaction_prompt()

    # 构建请求入参
    content = build_content(frames)
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]
    
    json_output, api_output = await retry_llm_call(messages, idx, interaction_calc_score)
    return json_output, api_output

async def interpenetration(frames, idx):
    ################# 穿模 #################
    # 创建prompt
    """中文：评估物体之间以及物体与机械臂之间的穿模问题。
English: Evaluate interpenetration between objects and between objects and robot arms."""
    interpenetration_system_prompt = create_interpenetration_prompt()
    
    content = build_content(frames)
    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": interpenetration_system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, interpenetration_calc_score)
    return json_output, api_output

async def shape(frames, idx):
    # 创建prompt
    """中文：评估物体形状结构与弹性变化是否合理。
English: Evaluate whether object structure and elastic deformation are plausible."""
    shape_system_prompt = create_shape_prompt()
    
    content = build_content(frames)
    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": shape_system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, shape_calc_score)
    return json_output, api_output

async def physics_reflection(frames, idx, interaction_output, interpenetration_output, shape_output):
    
    ################# reflection #################
    """中文：整合交互、穿模与形变判断，生成物理一致性的反思评分。
English: Combine interaction, interpenetration, and shape judgments into a reflected physics score."""
    reflection_system_prompt = create_physics_reflection_prompt()
    reflection_content = build_physics_reflection_content(frames, interaction_output, interpenetration_output, shape_output)
    # 构建请求入参
    messages = [
       {
           "role": "system",
           "content": reflection_system_prompt
       },
       {
           "role": "user",
           "content": reflection_content
       }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, physics_reflection_calc_score)
    return json_output, api_output

async def photometric_smoothness(frames, idx):
    # 创建prompt
    """中文：通过模型判断视频的光度平滑性。
English: Use the model to judge photometric smoothness of the video."""
    shape_system_prompt = create_photometric_smoothness_prompt()
    
    content = build_content(frames)
    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": shape_system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, color_calc_score)
    return json_output, api_output

async def physics(frames, idx):
    """中文：聚合交互、穿模、形变与反思模块，计算物理/运动一致性。
English: Aggregate interaction, interpenetration, shape, and reflection modules for physics/motion consistency."""
    interaction_json_output, interaction_output = await interaction(frames, idx)
    interpenetration_json_output, interpenetration_output = await interpenetration(frames, idx)
    shape_json_output, shape_output = await shape(frames, idx)
    physics_reflection_json_output, physics_reflection_out = await physics_reflection(
        frames,
        idx,
        interaction_json_output,
        interpenetration_json_output,
        shape_json_output,
    )
    return (interaction_json_output, interaction_output, 
            interpenetration_json_output, interpenetration_output, 
            shape_json_output, shape_output,
            physics_reflection_json_output, physics_reflection_out)

async def task_finish_state(description, frames, idx):
    # 创建prompt
    """中文：评估视频最终状态是否满足任务目标。
English: Evaluate whether the final video state satisfies the task goal."""
    system_prompt = create_task_finish_state_prompt()

    content = build_content(frames, description)

    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, task_calc_score)
    return json_output, api_output

async def eliminate_ambiguity(description, frames, idx):
    # 创建prompt
    """中文：评估视频是否消除了任务执行中的关键歧义。
English: Evaluate whether the video resolves key ambiguities in task execution."""
    system_prompt = create_eliminate_ambiguity_prompt()

    content = build_content(frames, description)

    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, task_calc_score)
    return json_output, api_output

async def task_complete(description, frames, idx):
    # 创建prompt
    """中文：综合过程与结果维度评估任务完成度。
English: Evaluate task completion from both process and outcome dimensions."""
    system_prompt = create_task_complete_prompt()

    content = build_content(frames, description)

    # 构建请求入参
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    json_output, api_output = await retry_llm_call(messages, idx, task_complete_calc_score)
    return json_output, api_output

# 多batcj+异步
async def process_one_video(video_path: str, description: str, idx: int):
    """中文：对单个视频运行完整 EWA-Reward 流程并返回结构化结果。
English: Run the full EWA-Reward pipeline for one video and return structured results."""
    raw_rgb_frames, openai_frames = extract_all_frames(video_path)
    all_frames_num = len(openai_frames)
    print('总帧数：', all_frames_num)

    # ===================== 【统一用字典存储】 =====================
    result = {
        "planning_json_output": None,
        "planning_api_output": None,
        "clarity_and_brightness_json_output": None,
        "clarity_and_brightness_api_output": None,
        "clarity_and_brightness_reflection_json_output": None,
        "clarity_and_brightness_reflection_api_output": None,
        "color_json_output": None,
        "color_api_output": None,
        "first_frame_json_output": None,
        "first_frame_api_output": None,
        "abnormal_frame_json_output": None,
        "abnormal_frame_api_output": None,
        "instruction_following_json_output": None,
        "instruction_following_api_output": None,
        "instruction_reflection_following_json_output": None,
        "instruction_reflection_following_api_output": None,
        "interaction_json_output": None,
        "interaction_output": None,
        "interpenetration_json_output": None,
        "interpenetration_output": None,
        "shape_json_output": None,
        "shape_output": None,
        "physics_reflection_json_out": None,
        "physics_reflection_out": None,
        "task_json_output": None,
        "task_api_output": None,
        "background_consistency_json_output": None,
        "background_consistency_api_output": None,
        "motion_quality_score": None,
        "total_score": 0
    }

    # ---------------------------------planning---------------------------------------
    frames = sample_uniform_frames(openai_frames, 20)
    result["planning_json_output"], result["planning_api_output"] = await planning(description, frames, idx)

    if result["planning_api_output"]["score"] == "Very Poor":
        result["total_score"] = 0
        return result

    elif result["planning_api_output"]["score"] == "Moderate Issues":
        planning_weight = 0.6
    else:
        planning_weight = 1

    # ------------------------ 1 清晰度+亮度 色彩 首帧一致 ------------------------------
    frames = sample_uniform_frames(openai_frames, 10)
    (result["clarity_and_brightness_json_output"],
    result["clarity_and_brightness_api_output"],
    result["clarity_and_brightness_reflection_json_output"],
    result["clarity_and_brightness_reflection_api_output"],
    result["color_json_output"],
    result["color_api_output"],
    result["first_frame_json_output"],
    result["first_frame_api_output"],
    result["abnormal_frame_json_output"],
    result["abnormal_frame_api_output"]) = await vision_quality(frames, idx)

    # 只要有一项不合格，直接返回最新结果
    # if (result["clarity_and_brightness_reflection_api_output"]['score'] == -1
    #     or result["first_frame_api_output"]['score'] == -1
    #     or result["color_api_output"]['score'] == -1
    #     or result["abnormal_frame_api_output"]['score'] == -1):
    #     result["total_score"] = -1
    #     return result

    if result["clarity_and_brightness_api_output"]['score'] == -1:
        result["total_score"] = -1
        return result
    
    # vision_score = (result["clarity_and_brightness_reflection_api_output"]['score']
    #                 * result["first_frame_api_output"]['score']
    #                 * result["color_api_output"]['score']
    #                 * result["abnormal_frame_api_output"]['score'])

    # vision_score = (result["clarity_and_brightness_reflection_api_output"]['score']
    #                 * result["first_frame_api_output"]['score'])

    vision_score = result["clarity_and_brightness_api_output"]['score']
    # print(vision_score)
    # if vision_score <= 0.1:
    # if vision_score < 1:
        # result["total_score"] = (vision_score * 0.2) * planning_weight
        # return result
    
    # ---------------------------- 2 动作质量 ------------------------------
    # 当前开源版本暂不启用额外动作质量指标；后续接入真实指标后再纳入总分。
    # Motion-quality metrics are disabled in the current release until a real
    # comparison metric is integrated.
    # result["motion_quality_score"] = <future real motion-quality metric>
    
    # ---------------------------- 3 物理/运动 ------------------------------
    frames = sample_uniform_frames(openai_frames, 64)
    (result["interaction_json_output"], result["interaction_output"],
    result["interpenetration_json_output"], result["interpenetration_output"],
    result["shape_json_output"], result["shape_output"],
    result["physics_reflection_json_out"], result["physics_reflection_out"]) = await physics(frames, idx)

    if result["physics_reflection_out"]['score'] == -1:
        result["total_score"] = (vision_score*0.2) * planning_weight
        return result

    physics_score = result["physics_reflection_out"]['score']
    if physics_score < 1:
        result["total_score"] = (vision_score*0.2+physics_score*0.3) * planning_weight
        return result
    
    # ---------------------------- 4 指令跟随 ------------------------------
    frames = sample_uniform_frames(openai_frames, 20)
    result["instruction_following_json_output"], result["instruction_following_api_output"] = await instruction_following(description, frames, idx)
    result["instruction_reflection_following_json_output"], result["instruction_reflection_following_api_output"] = await instruction_following_reflection(result["instruction_following_json_output"], frames, idx)
    result["background_consistency_json_output"], result["background_consistency_api_output"] = await background_consistency(description, frames, idx)

    if (result["instruction_reflection_following_api_output"]["score"] == -1
        or result["background_consistency_api_output"]['score'] == -1):
        result["total_score"] = (vision_score*0.2+physics_score*0.3) * planning_weight
        return result
    
    instruction_score = (0.4 * result["instruction_reflection_following_api_output"]["score"]
                    + 0.6 * result["background_consistency_api_output"]['score'])

    # ---------------------------- 5 任务完成度 ------------------------------
    result["task_json_output"], result["task_api_output"] = await task_complete(description, frames, idx)
    if result["task_api_output"]['score'] == -1:
        result["total_score"] = (vision_score*0.2+physics_score*0.3+instruction_score*0.2) * planning_weight
        return result

    task_score = result["task_api_output"]['score']
    result["total_score"] = (vision_score*0.2+physics_score*0.3+instruction_score*0.2+task_score*0.3) * planning_weight
    return result

async def process_one_video_safe(
    video_path: str,
    description: str,
    idx: int,
):
    """中文：带异常保护地评估单个视频，失败时返回标准错误结果。
English: Evaluate one video with exception protection and return a standard error result on failure."""
    try:
        return await process_one_video(
            video_path,
            description,
            idx,
        )
    except Exception as exc:
        traceback.print_exc()
        return {
            "planning_api_output": {"index": idx, "score": -1, "status": "error"},
            "total_score": -1,
            "error": str(exc),
        }

async def save_video_and_prompt(video_list: list[str], shared_prompt: str):
    """中文：使用全局日志目录保存本次请求的视频与 prompt。
English: Save request videos and prompt under the global log directory."""
    return await save_video_and_prompt_base(video_list, shared_prompt, BASE_ROOT)

class VideoRequest(BaseModel):
    """中文：/eval_video 接口的请求体结构。
English: Request body schema for the /eval_video endpoint."""
    video_path: List[str]
    prompt: str

def validate_request(data: VideoRequest) -> None:
    """中文：校验请求中的视频路径和任务 prompt 是否有效。
English: Validate video paths and task prompt in the incoming request."""
    if not data.video_path:
        raise HTTPException(status_code=400, detail="video_path must contain at least one video.")
    if not data.prompt or not data.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must be a non-empty string.")
    missing = [path for path in data.video_path if not Path(path).is_file()]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"message": "Some video paths do not exist.", "missing": missing},
        )

@app.get("/health")
async def health():
    """中文：返回服务健康状态和关键运行配置。
English: Return service health status and key runtime configuration."""
    return {
        "status": "ok",
        "model": SETTINGS.model,
        "api_base": SETTINGS.api_base,
        "motion_quality_enabled": SETTINGS.enable_motion_quality,
        "motion_quality_available": MOTION_QUALITY_AVAILABLE,
        "log_root": str(SETTINGS.log_root),
    }

@app.post("/eval_video")
async def eval_video(data: VideoRequest):  
    """中文：接收批量视频评估请求，并以 JSONL 流式返回 reward 分数。
English: Accept a batch video evaluation request and stream reward scores as JSONL."""
    print('接收到请求')
    validate_request(data)
    video_path = data.video_path
    description = data.prompt
    current_folder = None
    if SETTINGS.save_inputs:
        current_folder = await save_video_and_prompt(video_path, description)

    tasks = [
        process_one_video_safe(video_path[i], description, i)
        for i in range(len(video_path))
    ]

    async def stream_results():
        """中文：按完成顺序流式输出每个视频的评估结果。
English: Stream each video's evaluation result in completion order."""
        for coro in asyncio.as_completed(tasks):
            # 现在 await coro 返回的是【字典 result】，不是一长串元组！
            result = await coro

            # 直接把整个字典传给保存函数（超级简洁）
            if current_folder is not None and not result.get("error"):
                save_result_to_txt_table(current_folder, result)

            # 构造返回（直接从字典取值，不用一堆变量）
            total_score = result["total_score"]
            planning_api_output = result["planning_api_output"]

            if total_score != -1:
                return_api_output = {
                    "index": planning_api_output["index"],
                    "score": total_score,
                    "status": "success"
                }
            else:
                return_api_output = {
                    "index": planning_api_output["index"],
                    "score": -1,
                    "status": result.get("error") and "error" or "max_retry_failed"
                }
                if result.get("error"):
                    return_api_output["error"] = result["error"]
            yield json.dumps(return_api_output, ensure_ascii=False) + "\n"

    return StreamingResponse(stream_results(), media_type="application/jsonl; charset=utf-8")

# 启动 FastAPI + Uvicorn
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=SETTINGS.host,
        port=SETTINGS.port
    )
