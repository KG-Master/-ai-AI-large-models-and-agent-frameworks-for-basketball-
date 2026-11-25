import json
import os.path
import random

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

external_data = {}

rag = RagSummarizeService()


user_ids = ["1001","1002","1003","1004","1005","1006","1007","1008","1009","1010"]



@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)

@tool(description="获取指定城市的实时天气、气温、湿度与降雨概率，以纯字符串形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"

@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return random.choice(["北京", "上海", "广州", "深圳", "杭州"])

@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)

@tool(description="获取外部系统中存在记录的月份，以纯字符串形式返回")
def get_current_month() -> str:
    # 先确保外部数据已加载：首次调用时解析 records.csv 并缓存到全局 external_data，
    # 已加载过则直接复用缓存，不会重复读文件
    generate_external_data()
    
    # 从所有用户的所有记录中收集月份，做两层推导：
    #   for user in external_data.values()  -> 遍历每个用户
    #   for m in user                      -> 遍历该用户下的每个月份键
    # 外层用 {} 变成集合，自动去重（10 个用户都有 2025-01，只保留一个）
    # 再 sorted() 排序，保证返回的月份是按时间顺序可预期的
    months = sorted({m for user in external_data.values() for m in user})

    # 从真实存在的月份里随机取一个：
    # - 月份来自数据本身，而不是硬编码的 month_arr，所以不会取到库里没有的月份
    # - 数据为空（CSV 为空或缺列）时 months 为空列表，此时返回空字符串，
    #   避免 random.choice([]) 抛 IndexError，调用方也能识别“无可用月份”
    return random.choice(months) if months else ""


def generate_external_data():
    """
    {
        "user_id": {
            "month": {"特征": xxx, "进攻效率": xxx, ...}
            "month": {"特征": xxx, "进攻效率": xxx, ...}
            "month": {"特征": xxx, "进攻效率": xxx, ...}
            ...
        },
        "user_id": {
            "month": {"特征": xxx, "进攻效率": xxx, ...}
            "month": {"特征": xxx, "进攻效率": xxx, ...}
            "month": {"特征": xxx, "进攻效率": xxx, ...}
            ...
        },
        "user_id": {
            "month": {"特征": xxx, "进攻效率": xxx, ...}
            "month": {"特征": xxx, "进攻效率": xxx, ...}
            "month": {"特征": xxx, "进攻效率": xxx, ...}
            ...
        },
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                offensive_efficiency: str = arr[2].replace('"', "")
                equipment_and_physical_fitness: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "进攻效率": offensive_efficiency,
                    "装备和物理素质": equipment_and_physical_fitness,
                    "对比": comparison,
                }


@tool(description="获取指定球员在指定月份的月度训练与比赛记录，以纯字符串形式返回，未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return json.dumps(external_data[user_id][month], ensure_ascii=False)
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到球员：{user_id}在{month}的训练记录数据")
        return ""

if __name__ == '__main__':
    print(fetch_external_data("1001", "2025-01"))


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成场景动态注入上下文信息，为后续提示词切换提供上下文支撑")
def fill_context_for_report():
    return "fill_context_for_report已调用"
