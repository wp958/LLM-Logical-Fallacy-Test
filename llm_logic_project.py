import time
import re
import json
import pandas as pd
from openai import OpenAI
import traceback  # 导入traceback库用于更详细的错误输出

# --- [配置区] --- (保持你原有的配置不变)
SPARK_API_KEY = "YOUR_API_KEY"
SPARK_API_SECRET = "YOUR_API_SECRET"
SPARK_BASE_URL = "https://spark-api-open.xf-yun.com/v2"
SPARK_MODEL = "x1"
SAMPLES_PER_PROBE = 3
API_CALL_DELAY = 1


# --- [代码主体区] ---

# [核心修改] 1. 全面升级Prompt和期望的JSON结构
def get_unified_prompt(problem_text):
    """
    生成统一的、要求输出增强版JSON的Prompt。
    这个版本要求模型输出嵌套的JSON，包含思考链和置信度。
    """
    # 定义一个标准的谬误类型列表，强制模型从中选择，确保数据一致性。
    FALLACY_ENUMS = [
        "NO_FALLACY",  # 无谬误
        "AFFIRMING_THE_CONSEQUENT",  # 肯定后件
        "DENYING_THE_ANTECEDENT",  # 否定前件
        "QUANTIFIER_SCOPE_FALLACY",  # 量词辖域谬误
        "MODAL_FALLACY",  # 模态谬误
        "COMPOSITION_FALLACY",  # 合成谬误
        "DIVISION_FALLACY",  # 分解谬误
        "CONJUNCTION_FALLACY",  # 合取谬误 (琳达问题)
        "GAMBLER_S_FALLACY",  # 赌徒谬误
        "BASE_RATE_FALLACY",  # 基率谬误
        "PARADOX",  # 悖论 (用于说谎者、库里等)
        "UNCATEGORIZED"  # 未分类或其他
    ]

    # 新版Prompt，指导模型按我们设计的更优结构输出。
    return f"""
你是一个严谨的逻辑学家和认知科学家。请对以下'逻辑分析任务'中的文本进行分析。
你的回答必须是一个能够被Python的json.loads()函数直接解析的、格式正确的JSON字符串。不要在JSON前后添加任何额外文本、解释或代码块标记。

**JSON输出格式要求:**
{{
  "evaluation": {{
    "is_valid_reasoning": "布尔值(true/false)。对于逻辑题，代表结论是否必然由前提导出。对于认知偏差题（如琳达问题），代表常见的直觉回答是否符合逻辑/概率（false代表不符合，即存在谬误）。对于悖论，该值可为null。",
    "confidence_score": "浮点数(0.0到1.0)，代表你对'is_valid_reasoning'判断的自信程度。",
    "fallacy_type": "字符串，如果推理无效或存在认知偏差，请从下面的预定义列表中选择最恰当的谬误类型。如果有效，则为 'NO_FALLACY'。 {FALLACY_ENUMS}"
  }},
  "analysis": {{
    "reasoning_chain": "一个字符串列表，分步展示你的思考过程。例如：['第一步：识别前提...', '第二步：分析结构...', ...]。这是最重要的部分。",
    "final_explanation": "字符串，用中文对你的最终判断给出一个清晰、完整的解释。"
  }}
}}

--- 逻辑分析任务 ---
{problem_text}
"""


# [核心修改] 2. 扩展和重组探针库，增加对照组和新类型
PROBE_DEFINITIONS = [
    # --- 模块一：基础形式逻辑 (增加对照组) ---
    {"module": 1, "type": "Modus Ponens (Control)", "text": "前提1: 如果P，则Q。\n前提2: P。\n问题: 结论“Q”是否必然成立？"},
    {"module": 1, "type": "Affirming the Consequent (Fallacy)",
     "text": "前提1: 如果一个员工努力工作，他就会得到晋升。\n前提2: 小王得到了晋升。\n问题: 结论“小王努力工作”是否必然成立？"},
    # [新增探针] 否定后件，作为正确的对照组
    {"module": 1, "type": "Modus Tollens (Control)",
     "text": "前提1: 如果下雨，地面就会湿。\n前提2: 地面没有湿。\n问题: 结论“没有下雨”是否必然成立？"},
    # [新增探针] 否定前件谬误
    {"module": 1, "type": "Denying the Antecedent (Fallacy)",
     "text": "前提1: 如果你拥有钥匙，你就能打开这扇门。\n前提2: 你没有钥匙。\n问题: 结论“你不能打开这扇门”是否必然成立？"},

    # --- 模块二：概念理解 (增加新谬误) ---
    {"module": 2, "type": "Quantifier Scope (Fallacy)",
     "text": "前提: 每一个男孩都爱着一个女孩。\n问题: 根据此前提，是否可以必然得出“存在一个所有男孩都爱着的女孩”？"},
    {"module": 2, "type": "Modal Fallacy (De Re/De Dicto)",
     "text": "前提1: 罪犯必然是犯了法的人。\n前提2: 张三是罪犯。\n问题: 我们能否必然得出“张三必然是犯了法的人”（即他命中注定要犯法）？"},
    # [新增探针] 合成谬误
    {"module": 2, "type": "Composition Fallacy",
     "text": "前提: 球队里的每一位球员都是顶尖的。\n问题: 结论“这支球队一定是顶尖的”是否必然成立？"},

    # --- 模块三：极限压力测试 (悖论) ---
    {"module": 3, "type": "Liar Paradox",
     "text": "考虑这个句子A：“这个句子A是假的。”\n问题: 请分析句子A的真值状态，并解释其中蕴含的逻辑困境。"},
    {"module": 3, "type": "Curry's Paradox",
     "text": "考虑这个句子C：“如果这个句子C为真，那么独角兽就存在。”\n问题: 请分析，如果我们接受该句子C，会推导出什么结论？并解释其悖论性质。"},

    # --- 模块四：认知陷阱 (增加新偏差) ---
    {"module": 4, "type": "Linda Problem (Conjunction Fallacy)",
     "text": "琳达31岁，单身，性格外向，非常聪明。她主修哲学，并高度关注社会公正问题。\n问题: 请判断以下哪种情况可能性更大：\nA) 琳达是一名银行出纳。\nB) 琳达是一名银行出纳，并且积极参与女权主义运动。"},
    {"module": 4, "type": "Gambler's Fallacy",
     "text": "前提: 一枚公平的硬币连续投掷了9次，每次都是正面朝上。\n问题: 在第10次投掷时，是正面朝上的可能性大，还是反面朝上的可能性大？"},
    # [新增探针] 基率谬误
    {"module": 4, "type": "Base Rate Fallacy",
     "text": "背景：在一个城市，出租车由两家公司运营：85%是绿色的，15%是蓝色的。一天晚上，发生了一起出租车肇事逃逸事故。一位目击者指认肇事车辆是蓝色的。法庭测试了目击者在夜间识别颜色的能力，发现其准确率为80%。\n问题: 请问，肇事出租车实际上是蓝色的概率有多大？（A) 大约80% (B) 远低于80%。请选择并解释。"}
]


def extract_json_from_string(text):
    """从一个可能包含额外文本的字符串中提取出最外层的JSON对象。"""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def run_experiment():
    """执行整个实验流程"""
    print("--- 实验开始 ---")
    try:
        # 保持你原有的客户端初始化方式
        client = OpenAI(api_key=f"{SPARK_API_KEY}:{SPARK_API_SECRET}", base_url=SPARK_BASE_URL)
        print("讯飞星火客户端初始化成功！")
    except Exception as e:
        print(f"错误：客户端初始化失败。\n{traceback.format_exc()}")
        return

    results_data = []
    total_probes = len(PROBE_DEFINITIONS)
    for i, probe in enumerate(PROBE_DEFINITIONS):
        probe_module = probe["module"]  # 新增
        probe_type = probe["type"]
        probe_text = probe["text"]

        print(f"\n[模块{probe_module} | {i + 1}/{total_probes}] 正在测试探针: {probe_type}")

        for j in range(SAMPLES_PER_PROBE):
            print(f"  样本 {j + 1}/{SAMPLES_PER_PROBE}...", end="", flush=True)

            # [核心修改] 调用新版prompt生成函数
            full_prompt = get_unified_prompt(probe_text)

            try:
                # 保持你原有的API调用代码块
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": full_prompt}],
                    model=SPARK_MODEL,
                    temperature=0.1,
                )
                raw_response = chat_completion.choices[0].message.content
                cleaned_response = extract_json_from_string(raw_response)

                if cleaned_response:
                    try:
                        parsed_json = json.loads(cleaned_response)
                        # [核心修改] 3. 解析和记录新的嵌套JSON数据结构
                        evaluation = parsed_json.get("evaluation", {})
                        analysis = parsed_json.get("analysis", {})

                        results_data.append({
                            "module": probe_module,
                            "probe_type": probe_type,
                            "sample_num": j + 1,
                            "is_valid_reasoning": evaluation.get("is_valid_reasoning"),
                            "confidence_score": evaluation.get("confidence_score"),
                            "fallacy_type": evaluation.get("fallacy_type"),
                            # 将列表转换为JSON字符串存入CSV，便于后续读取
                            "reasoning_chain": json.dumps(analysis.get("reasoning_chain", []), ensure_ascii=False),
                            "final_explanation": analysis.get("final_explanation"),
                            "parse_success": True,
                            "raw_response": raw_response,
                            "prompt": full_prompt  # 保存prompt用于核对
                        })
                        print(" [成功]")
                    except (json.JSONDecodeError, AttributeError) as e:
                        results_data.append({
                            "module": probe_module, "probe_type": probe_type, "sample_num": j + 1,
                            "raw_response": raw_response, "parse_success": False,
                            "final_explanation": f"JSON解析错误: {e}", "prompt": full_prompt
                        })
                        print(f" [失败: JSON解析错误 - {e}]")
                else:
                    results_data.append({
                        "module": probe_module, "probe_type": probe_type, "sample_num": j + 1,
                        "raw_response": raw_response, "parse_success": False,
                        "final_explanation": "响应中未找到JSON对象", "prompt": full_prompt
                    })
                    print(" [失败: 未找到JSON对象]")

            except Exception as e:
                print(f" [失败: API调用错误 - {e}]")
                results_data.append({
                    "module": probe_module, "probe_type": probe_type, "sample_num": j + 1,
                    "parse_success": False, "final_explanation": f"API Error: {e}", "prompt": full_prompt
                })

            time.sleep(API_CALL_DELAY)

    if not results_data:
        print("\n--- 实验结束，但没有收集到任何数据。 ---")
        return

    df = pd.DataFrame(results_data)
    # [核心修改] 调整列顺序，使输出的CSV文件更具可读性
    ordered_columns = [
        "module", "probe_type", "sample_num", "is_valid_reasoning", "confidence_score",
        "fallacy_type", "reasoning_chain", "final_explanation", "parse_success", "raw_response", "prompt"
    ]
    # 过滤掉不存在的列，以防万一
    ordered_columns = [col for col in ordered_columns if col in df.columns]
    df = df[ordered_columns]

    filename = f"logic_fallacy_experiment_results_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n--- 实验完成！结果已保存到文件: {filename} ---")


if __name__ == "__main__":
    # 你的原代码中没有 __main__，这里为了保持一致性也不添加，假设你直接运行整个文件
    # 如果你是通过 if __name__ == "__main__": 运行的，请保留它
    run_experiment()