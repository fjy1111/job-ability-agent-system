import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    防止大模型返回 ```json ... ``` 或前后带解释文字。
    """
    if not text:
        return {}

    text = text.strip()

    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    return {}


def fallback_top5_gap_paths(job_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    大模型失败时的兜底逻辑：根据已有 skill_gaps 给每个岗位生成简单路径。
    """
    result = []

    for job in job_recommendations[:5]:
        job_name = job.get("job_name", "未知岗位")
        skill_gaps = job.get("skill_gaps", [])

        result.append({
            "job_name": job_name,
            "gap_list": skill_gaps,
            "recommended_projects": [
                f"{job_name}相关小型练习项目",
                f"{job_name}方向综合实战项目"
            ],
            "learning_stages": [
                {
                    "stage": "第一阶段：基础补强",
                    "duration": "第1-2个月",
                    "goal": f"补齐{job_name}岗位所需的基础技能",
                    "actions": [
                        f"系统学习岗位短板技能：{'、'.join(skill_gaps) if skill_gaps else '岗位基础技能'}",
                        "整理学习笔记，完成基础练习",
                        "结合目标岗位要求，建立技能清单"
                    ],
                    "deliverables": [
                        "一份岗位技能差距清单",
                        "一份基础学习笔记",
                        "若干基础练习代码或文档"
                    ]
                },
                {
                    "stage": "第二阶段：项目实践",
                    "duration": "第3-4个月",
                    "goal": f"完成一个与{job_name}匹配的项目作品",
                    "actions": [
                        "选择一个岗位相关项目进行完整实践",
                        "完成需求分析、实现过程和结果总结",
                        "将项目整理到简历或作品集中"
                    ],
                    "deliverables": [
                        "一个完整项目作品",
                        "一份项目说明文档",
                        "一份可写入简历的项目经历"
                    ]
                },
                {
                    "stage": "第三阶段：就业准备",
                    "duration": "第5-6个月",
                    "goal": f"提升{job_name}岗位投递竞争力",
                    "actions": [
                        "根据岗位要求修改简历",
                        "准备常见面试问题",
                        "进行模拟面试和复盘"
                    ],
                    "deliverables": [
                        "一份岗位定制简历",
                        "一份面试题复盘文档",
                        "一份投递岗位清单"
                    ]
                }
            ]
        })

    return result


def generate_top5_gap_paths(
    student_data: Dict[str, Any],
    job_recommendations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    为 TOP5 每个岗位生成：
    1. 岗位差距清单
    2. 推荐项目
    3. 个性化补齐路径
    """

    # 你这里根据自己用的模型接口改。
    # 如果你现在已经有 OpenAI / DeepSeek / 通义千问调用代码，
    # 只需要把 prompt 换成下面这个即可。

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

    prompt = f"""
你是大学生就业能力诊断系统中的岗位成长路径规划智能体。

下面是学生信息：
{json.dumps(student_data, ensure_ascii=False)}

下面是系统已经计算出的 TOP5 岗位推荐结果：
{json.dumps(job_recommendations[:5], ensure_ascii=False)}

请你必须为 TOP5 中的每一个岗位都生成个性化成长路径，不能只生成第一个岗位。

返回严格 JSON，不要 Markdown，不要解释文字。

字段要求如下：

{{
  "top5_gap_paths": [
    {{
      "job_name": "岗位名称，必须和输入中的 job_name 完全一致",
      "gap_list": ["该岗位当前最主要的差距1", "差距2", "差距3"],
      "recommended_projects": ["推荐项目1", "推荐项目2"],
      "learning_stages": [
        {{
          "stage": "第一阶段：基础补强",
          "duration": "第1-2个月",
          "goal": "阶段目标",
          "actions": ["行动任务1", "行动任务2", "行动任务3"],
          "deliverables": ["验收成果1", "验收成果2"]
        }},
        {{
          "stage": "第二阶段：项目实践",
          "duration": "第3-4个月",
          "goal": "阶段目标",
          "actions": ["行动任务1", "行动任务2", "行动任务3"],
          "deliverables": ["验收成果1", "验收成果2"]
        }},
        {{
          "stage": "第三阶段：就业准备",
          "duration": "第5-6个月",
          "goal": "阶段目标",
          "actions": ["行动任务1", "行动任务2", "行动任务3"],
          "deliverables": ["验收成果1", "验收成果2"]
        }}
      ]
    }}
  ]
}}

要求：
1. top5_gap_paths 的长度必须等于输入岗位数量，最多 5 个。
2. 每个岗位都要有自己的 gap_list、recommended_projects、learning_stages。
3. 不要只围绕匹配度最高的岗位生成。
4. gap_list 要结合该岗位的 skill_gaps 和学生已有技能。
5. recommended_projects 要贴合岗位方向。
6. learning_stages 必须是三个阶段。
"""

    try:
        # ====== 这里替换成你自己原来的大模型调用方式 ======
        # 下面这个只是示例结构，不能直接凭空调用。
        #
        # response_text = call_llm(prompt)
        #
        # parsed = safe_json_loads(response_text)

        parsed = {}

        # 如果你暂时还没有接好大模型，就先走兜底
        if not parsed:
            parsed = {
                "top5_gap_paths": fallback_top5_gap_paths(job_recommendations)
            }

        top5_gap_paths = parsed.get("top5_gap_paths", [])

        # 防止大模型少生成，自动补齐到 TOP5
        if len(top5_gap_paths) < len(job_recommendations[:5]):
            existed_names = {item.get("job_name") for item in top5_gap_paths if isinstance(item, dict)}
            fallback_items = fallback_top5_gap_paths(job_recommendations)

            for item in fallback_items:
                if item.get("job_name") not in existed_names:
                    top5_gap_paths.append(item)

        parsed["top5_gap_paths"] = top5_gap_paths[:5]
        parsed["used_llm"] = bool(api_key)

        return parsed

    except Exception as e:
        return {
            "top5_gap_paths": fallback_top5_gap_paths(job_recommendations),
            "used_llm": False,
            "agent_warning": str(e)
        }