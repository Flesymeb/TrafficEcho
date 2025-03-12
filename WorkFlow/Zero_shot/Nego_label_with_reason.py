import json
import logging
import os
from copy import deepcopy

import requests

# 配置通义千问API
API_KEY = "sk-e82535fe61064496860ac3fe9c5f6d7d"  # 替换为您的实际API密钥
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"  # 更新为正确的API URL

# 定义指标集
categories = ["安全性", "运行效率", "可靠性", "舒适性", "便捷性", "信息化与智能化", "经济性", "环境友好性", "社会可持续性"]

# 全局变量用于记录总的 token 使用量
total_token = 0

# 设置全局日志文件
log_file = "logs/negotiation_process.log"
os.makedirs("logs", exist_ok=True)

def setup_logger():
    """配置日志，增强指标标识"""
    logger = logging.getLogger("negotiation_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter(
            "处理时间 %(asctime)s \n%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()


def call_qwen_api(messages, timeout=40, Model="qwen-max"):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": Model,
        "messages": messages,
        "max_tokens": 1024,
    }
    try:
        response = requests.post(
            API_URL, json=payload, headers=headers, timeout=timeout
        )
        if response.status_code == 200:
            response_data = response.json()
            result = (
                response_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            usage = response_data.get("usage", {}).get("total_tokens", 0)
            return result, usage
        else:
            raise Exception(
                f"API请求失败，状态码：{response.status_code}, 响应内容：{response.text}"
            )
    except requests.exceptions.Timeout:
        print("API请求超时，请检查网络连接或增加超时时间。")
        return None, 0
    except Exception as e:
        print(f"API请求异常：{e}")
        return None, 0


def generator_label(weibo_content, feedback=""):
    prompt = (
        f"## 你是一个专业的交通评价分析师，你的任务是对用户提供的微博内容进行主观评价维度上的分类，并简要解释理由。评价维度分类指标包括:{categories}。\n"
        f"## 以下是对分类指标的一些解释:\n"
        f"经济性（Economic Efficiency）：涉及到花钱、票价、燃油费等关于出行的花费的评价、涉及到运营成本、投资回报率的就是经济性相关；\n"
        f"安全性（Safety）：涉及到交通事故、危险驾驶（超车加塞等）、交通运营安全、设施安全性、乘客安全、运行环境安全、应急响应、事故预防能力（比如查酒驾、交警检查、交通管制等，但只涉及到交警等不算）的，就是安全性相关\n"
        f"但是需要注意的是，如果查酒驾导致了更加拥堵，那就不涉及到安全性，而是运行效率\n"
        f"运行效率（Operational Efficiency）：涉及到道路上的拥堵情况(不是公共交通拥挤情况)、公共交通的运行时间和等待时间的就是运行效率相关；\n"
        f"可靠性（Reliability）：涉及到公共交通准点率、运行时间的稳定性（比如晚点、早点之类的）、服务一致性、线路可靠性、交通事件对运营的影响的、司机拒载等，还有道路的修缮保养情况，提到道路平坦宽敞干净就是可靠性相关，常发性道路拥堵一般不算可靠性；\n"
        f"舒适性（Comfort）：涉及到驾驶/乘坐舒适度、公共交通拥挤、车厢整洁度、设备情况、通风、温度等的或者其他超出交通出行的最基础服务的，就是舒适性相关，也包括类似于堵车对司机或乘客产生的不适感、路况不好带来的不适感；公共交通没有厕所也是舒适性等等；\n"
        f"便捷性（Convenience）：涉及到交通方式的可获得性、线路或路网的覆盖率、换乘便捷性、票务便利性、出行时间成本、无障碍出行等的，是便捷性相关，不要看到“方便”或“便捷”两个字就盲目直接分类到便捷性；\n"
        f"信息化与智能化（Information & Smart Technology）：涉及到交通信息的可获取性、实时信息服务、交通系统的智能化程度、交通预测、智能驾驶、导航系统、交通服务APP方面的，就是信息化与智能化相关；\n"
        f"环境友好性（Environmental Friendliness）：涉及到新能源汽车在环境友好方面的优势、排放污染、绿色出行感受、交通基础设施建设对环境产生影响等的，就是环境友好性相关，注意只提到景色漂亮的内容不算环境友好性；\n"
        f"社会可持续性（Social Sustainability）：涉及到交通公平性、社会福祉、交通对于城市空间的影响、社会服务与责任的、以及对交通政策、交通规划、路口/道路设计、车道划分等提出的意见、要求整改、向企业或者有关部门提出建议的相关评论，就是社会可持续性相关。\n"
        f"请注意：请完整阅读微博正文，将评论内容分类到一个或几个合适的评价维度。但不要过度解读，只判断与交通和出行有关的内容，不要受到其余非交通的内容的干扰，以免使一些类别没有必要，比如此次出行的目的（游玩的开心、上班的绝望、购物方面等等）、沿途的风景等等。\n"
        f"## 微博内容：\n"
        f"{weibo_content}\n\n"
        f"## 反馈机制：\n"
        f"如果存在 feedback，说明你的上一轮分类未被判别器D认可，未达到共识。\n"
        f"请结合以下 feedback 信息优化你的分类结果与理由：\n"
        f"### 上一轮 feedback 信息： {feedback}\n\n"
        f"## 返回格式要求：\n"
        f"请严格按照以下格式返回一个JSON对象结果：\n"
        f'{{\"类别\": [\"安全性\",\"便捷性\"], \"理由\": \"你的理由\"}}'
    )

    messages = [{"role": "system", "content": prompt}]
    result, tokens_used = call_qwen_api(messages, Model="qwen-max")
    result = result.strip("```json").strip("```")

    try:
        response = json.loads(result)
        G_label = response.get("类别", [])
        reason = response.get("理由", "").strip()

        if not all(label in categories for label in G_label):
            raise ValueError("分类结果包含未知类别")

        return G_label, reason, tokens_used
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"生成器G返回的格式错误：{result}，错误详情：{e}")

def discriminator_evaluate(original_content, generator_label, generator_reason, feedback=""):
    prompt = (
        f"## 你是一个专业的交通评价分析师，仅根据微博的内容，请评估生成器G关于该微博内容在主观评价维度上的分类是否合理。\n"
        f"## 类别列表：{categories}\n"
        f"以下是对分类指标的一些解释:\n"
        f"经济性（Economic Efficiency）：涉及到花钱、票价、燃油费等关于出行的花费的评价、涉及到运营成本、投资回报率的就是经济性相关；\n"
        f"安全性（Safety）：涉及到交通事故、危险驾驶（超车加塞等）、交通运营安全、设施安全性、乘客安全、运行环境安全、应急响应、事故预防能力（比如查酒驾、交警检查、交通管制等，但只涉及到交警等不算）的，就是安全性相关\n"
        f"但是需要注意的是，如果查酒驾导致了更加拥堵，那就不涉及到安全性，而是运行效率\n"
        f"运行效率（Operational Efficiency）：涉及到道路上的拥堵情况(不是公共交通拥挤情况)、公共交通的运行时间和等待时间的就是运行效率相关；\n"
        f"可靠性（Reliability）：涉及到公共交通准点率、运行时间的稳定性（比如晚点、早点之类的）、服务一致性、线路可靠性、交通事件对运营的影响的、司机拒载等，还有道路的修缮保养情况，提到道路平坦宽敞干净就是可靠性相关，常发性道路拥堵一般不算可靠性；\n"
        f"舒适性（Comfort）：涉及到驾驶/乘坐舒适度、公共交通拥挤、车厢整洁度、设备情况、通风、温度等的或者其他超出交通出行的最基础服务的，就是舒适性相关，也包括类似于堵车对司机或乘客产生的不适感、路况不好带来的不适感；公共交通没有厕所也是舒适性等等；\n"
        f"便捷性（Convenience）：涉及到交通方式的可获得性、线路或路网的覆盖率、换乘便捷性、票务便利性、出行时间成本、无障碍出行等的，是便捷性相关，不要看到“方便”或“便捷”两个字就盲目直接分类到便捷性；\n"
        f"信息化与智能化（Information & Smart Technology）：涉及到交通信息的可获取性、实时信息服务、交通系统的智能化程度、交通预测、智能驾驶、导航系统、交通服务APP方面的，就是信息化与智能化相关；\n"
        f"环境友好性（Environmental Friendliness）：涉及到新能源汽车在环境友好方面的优势、排放污染、绿色出行感受、交通基础设施建设对环境产生影响等的，就是环境友好性相关，注意只提到景色漂亮的内容不算环境友好性；\n"
        f"社会可持续性（Social Sustainability）：涉及到交通公平性、社会福祉、交通对于城市空间的影响、社会服务与责任的、以及对交通政策、交通规划、路口/道路设计、车道划分等提出的意见、要求整改、向企业或者有关部门提出建议的相关评论，就是社会可持续性相关。\n\n"
        f"## 请注意，生成器G在分类时，有可能对一些内容过度解读，或受到并非交通相关的评价内容的干扰，导致有的类别可能没有必要，比如“今天坐公交，窗外景色好漂亮”，这不是环境友好性。\n"
        f"如果你认为分类合理，请返回“同意”；如果你认为分类不合理，请返回“不同意”，并提供优化建议。并请你至少保留生成器G的一个最合适的类别，不要全盘否定\n\n"
        f"## 微博内容：\n"
        f"{original_content}\n\n"
        f"## 生成器G的分类结果：\n"
        f"- **分类**：{generator_label}\n"
        f"- **理由**：{generator_reason}\n\n"
        f"## 反馈机制：\n"
        f"如果存在 feedback，说明上一轮协商未达到共识，你需要结合以下反馈信息优化你的评估：\n"
        f"### 上一轮 feedback 信息： {feedback}\n\n"
        f"## 返回格式要求：\n"
        f"请严格按照以下格式返回一个JSON对象结果：\n"
        f'{{\"评估\": \"同意/不同意\", \"理由\": \"你的理由\"}}'
    )

    messages = [{"role": "system", "content": prompt}]
    result, tokens_used = call_qwen_api(messages, Model="qwen-plus")
    result = result.strip("```json").strip("```")

    try:
        response = json.loads(result)
        evaluation = response.get("评估", "").strip()
        reason = response.get("理由", "").strip()

        if evaluation not in ["同意", "不同意"]:
            raise ValueError("评估结果不在预期范围内")

        return evaluation, reason, tokens_used
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"判别器D返回的格式错误：{result}，错误详情：{e}")


def negotiate_label(weibo_content, max_rounds=3, tolerance=0.1):
    global total_token
    negotiation_log = [f"正在处理"]
    final_round = 0
    feedback = ""  # 初始化反馈为空
    prev_label = None  # 记录上一轮的极性

    for round_num in range(1, max_rounds + 1):
        final_round = round_num

        try:
            # 生成器预测
            g_label, g_reason, g_tokens = generator_label(
                weibo_content, feedback
            )
            negotiation_log.append(
                f"第{round_num}轮生成器G\n"
                f"类别: {g_label}\n"
                f"理由: {g_reason}"
            )
            total_token += g_tokens
        except Exception as e:
            logger.error(f"G错误|{str(e)}")
            return None

        try:
            # 判别器评估
            d_agreement, d_reason, d_tokens = discriminator_evaluate(
                weibo_content, g_label, g_reason, feedback
            )
            negotiation_log.append(
                f"第{round_num}轮判别器D\n"
                f"反馈: {d_agreement}\n"
                f"理由: {d_reason}"
            )
            total_token += d_tokens
        except Exception as e:
            logger.error(f"D错误{str(e)}")
            return None

        # 达成共识
        if d_agreement == "同意":
            negotiation_log.append(f"达成共识（第{round_num}轮）")
            logger.info("\n".join(negotiation_log))
            return g_label, g_reason, final_round

        # 极性稳定检查
        if prev_label == g_label:
            negotiation_log.append(f"极性稳定终止（{g_label}）")
            logger.warning("\n".join(negotiation_log))
            return g_label, g_reason, final_round

        prev_label = g_label
        feedback = f"D反馈|{d_reason}"

        # 最大轮次终止
    negotiation_log.append(f"达到最大轮次（{max_rounds}）")
    logger.warning("\n".join(negotiation_log))
    return g_label, g_reason, final_round


if __name__ == "__main__":
    # 配置目录
    input_file = "merged_data_test_0.json"
    output_file = "classified_data_test_0.json"

    logger.info(f"======== 开始处理 ========")
    print(f"开始处理")
    # 读取 JSON 数据
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            weibo_list = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"读取 {input_file} 失败: {e}")
        print(f"读取 {input_file} 失败: {e}")

    new_weibo_list = deepcopy(weibo_list)  # 复制数据，避免污染原数据

    # 创建类别对应的输出文件
    category_output = {}

    for idx, weibo_data in enumerate(new_weibo_list):
        weibo_content = weibo_data.get("微博正文", "")
        logger.info(f"\n======== 处理第 {idx + 1} 条微博 ========")

        valid_data = True  # 标记当前微博数据是否有效

        result = negotiate_label(weibo_content)

        if result is None:
            logger.warning(f"跳过第 {idx + 1} 条微博，因协商错误")
            print(f"跳过第 {idx + 1} 条微博，因协商错误")
            valid_data = False

        # 仅在协商有效时，更新 JSON
        if valid_data:
            final_label, reasoning, iterations = result
            logger.info(f"最终分类：{final_label}, 协商轮次：{iterations}, 理由：{reasoning}")
            weibo_data["predicted_label"] = final_label  # 修改为极性字段
        else:
            weibo_data["predicted_label"] = []


    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_weibo_list, f, ensure_ascii=False, indent=4)
        logger.info(f"处理完成\n")
    except Exception as e:
        logger.error(f"写入失败: {e}")
        print(f"写入失败: {e}")

    # 输出总的 token 使用量
    print(f"所有微博数据处理完毕，共使用了 {total_token} 个 token")
    print(f"分类完成，结果已保存至", output_file)
