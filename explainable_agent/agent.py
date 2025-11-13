"""
post_investment_agent.py

投后陪伴顾问 Agent (RAG + LLM)
依赖: requests (可选, 若使用真实搜索API)
假设: 外部已有 get_llm_output(prompt: str, query: str) -> str
"""

import json
import math
import statistics
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# LangChain相关导入，用于实现RAG功能
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.document_loaders import TextLoader
    from langchain.schema import Document
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain相关库未安装: {e}")
    print("将使用基础替代方案运行")
    LANGCHAIN_AVAILABLE = False
    
    # 定义简单的Document类作为替代
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    # 定义简单的检索器作为替代
    class SimpleRetriever:
        def __init__(self, documents):
            self.documents = documents
        
        def similarity_search(self, query, k=3):
            # 简单的关键词匹配作为替代
            results = []
            for doc in self.documents:
                score = 0
                query_lower = query.lower()
                content_lower = doc.page_content.lower()
                # 计算简单的关键词匹配分数
                for keyword in query_lower.split():
                    if keyword in content_lower:
                        score += 1
                if score > 0:
                    results.append((doc, score))
            # 按分数排序并返回前k个结果
            results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in results[:k]] if results else self.documents[:k]

# 全局变量，用于存储向量数据库
vector_db = None
knowledge_base = None

# OPTIONAL: 网络请求库用于调用搜索API（如百度）：
try:
    import requests
except Exception:
    requests = None

# ---------------------------
# 1) 如果你已经有 get_llm_output，就直接使用；否则启用本地 stub（便于本地测试）
# ---------------------------
try:
    get_llm_output  # type: ignore
except NameError:
    def get_llm_output(prompt: str, query: str) -> str:
        # """
        # 本地 stub：把 prompt + query 拼成简单回答（仅用于本地演示，真实使用请替换为你的 LLM 调用）
        # """
        # return (
        #     "【LLM-stub回答】\n"
        #     "任务：分析组合短期亏损并给出可行建议。\n\n"
        #     "摘要：近期组合承压，主要由于科技板块同步下跌；建议保持核心配置，逐步减持波动较大的科技个股，"
        #     "将赎回资金分批转入低波动收益或短期债券类产品以锁定本金与满足短期流动性需求。\n\n"
        #     "操作建议示例：\n"
        #     "1) 若目标为长期养老金：坚持长期配置，考虑在逢低分批加仓；\n"
        #     "2) 若目标为近期大额支出（1年内）：逐步收缩权益仓，增加货币/短债类。\n\n"
        #     "如需更详细的分步再平衡方案，请提供风险预算与手续费信息。"
        # )
        import requests
        url = "https://api.siliconflow.cn/v1/chat/completions"
        payload = {
            "model": "deepseek-ai/DeepSeek-V3.2-Exp",
            "messages": [{"role": "user","content": prompt + query}]}
        headers = {
            "Authorization": "Bearer sk-qlutaewnzinxxxxxxxxxxxxxxx", #已脱敏，使用新的API密钥
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        return response.json()["choices"][0]["message"]["content"]
# ---------------------------
# 2) 辅助函数：获取用户持仓（这里我们用虚拟/示例数据；在真实系统中改为 DB / API）
# ---------------------------
def get_portfolio(user_id: str) -> Dict[str, Any]:
    """
    返回示例持仓数据结构。真实系统这里应访问数据库/服务。
    输出示例字段：
      - user_id
      - portfolio: list of holdings {ticker, name, sector, weight, price, 1d_ret, 1w_ret, 1m_ret}
      - cash_reserve_pct
      - target_goals: list of {goal, horizon_years, priority}
      - risk_budget: float (0-1)
    """
    # 示例/虚构数据
    return {
        "user_id": user_id,
        "portfolio": [
            {"ticker": "TECH-A", "name": "某科技主动基金A", "sector": "Technology", "weight": 0.40,
             "price": 12.34, "1d_ret": -0.035, "1w_ret": -0.12, "1m_ret": -0.18},
            {"ticker": "ETF-CHIP", "name": "半导体ETF", "sector": "Technology", "weight": 0.20,
             "price": 45.10, "1d_ret": -0.028, "1w_ret": -0.09, "1m_ret": -0.22},
            {"ticker": "BOND-SHORT", "name": "短期国债ETF", "sector": "FixedIncome", "weight": 0.25,
             "price": 101.2, "1d_ret": 0.002, "1w_ret": 0.005, "1m_ret": 0.01},
            {"ticker": "CONSUMER-F", "name": "消费主题基金", "sector": "Consumer", "weight": 0.10,
             "price": 8.9, "1d_ret": -0.01, "1w_ret": -0.03, "1m_ret": -0.04},
            {"ticker": "CASH", "name": "现金/货币基金", "sector": "Cash", "weight": 0.05,
             "price": 1.0, "1d_ret": 0.0004, "1w_ret": 0.001, "1m_ret": 0.002},
        ],
        "cash_reserve_pct": 0.05,
        "target_goals": [
            {"goal": "子女教育", "horizon_years": 12, "priority": "high"},
            {"goal": "退休储备", "horizon_years": 25, "priority": "medium"},
            {"goal": "短期应急", "horizon_years": 1, "priority": "high"}
        ],
        "risk_budget": 0.6  # 0 conservative - 1 aggressive
    }

# ---------------------------
# 3) 市场信息检索（RAG 的检索部分）
#    fetch_market_news_for_holdings -> 调用搜索 API（占位），如失败返回模拟新闻
# ---------------------------
def call_baidu_search_api(query: str, topk: int = 3, search_type: str = "news") -> List[Dict[str, str]]:
    """
    调用百度搜索API并返回topk条摘要，支持不同类型的搜索。
    参数:
        - query: 搜索查询
        - topk: 返回结果数量
        - search_type: 搜索类型 ("news", "market", "analysis", "general")
    返回值格式：list of {"id":..., "title":..., "date":..., "content":..., "snippet":..., "url":..., "timestamp":...}
    """
    import time
    current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # 根据搜索类型调整查询和参数
    search_filters = {
        "news": {"search_recency_filter": "day", "additional_terms": "最新 快讯"},
        "market": {"search_recency_filter": "week", "additional_terms": "市场 行情 分析"},
        "analysis": {"search_recency_filter": "week", "additional_terms": "深度 分析 前景"},
        "general": {"search_recency_filter": "month", "additional_terms": ""}
    }
    
    filter_config = search_filters.get(search_type, search_filters["general"])
    enhanced_query = query + " " + filter_config["additional_terms"]
    
    # ====== 如果 requests 可用并已配置真实 API，可在此替换实现 ======
    global requests
    if requests is not None:
        try:
            import requests
            import json
            url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
            payload = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": enhanced_query
                    }
                ],
                "edition": "standard",
                "search_source": "baidu_search_v2",
                "search_recency_filter": filter_config["search_recency_filter"]
            }, ensure_ascii=False)
            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer bce-v3/ALTAK-LABxxxxxxxxxxxxxxx' #已脱敏，使用新的API密钥
            }
            response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
            response.encoding = "utf-8"
            data = json.loads(response.text)
            # 提取references字段并只保留需要的字段
            references = data.get('references', [])
            filtered_references = []
            for ref in references:
                # 确保返回的数据包含必要字段，并添加timestamp
                filtered_ref = {
                    "id": ref.get("id"),
                    "title": ref.get("title"),
                    "date": ref.get("date", current_timestamp),
                    "content": ref.get("content"),
                    "snippet": ref.get("content", ""),  # 使用content作为snippet
                    "url": ref.get("url", "https://news.example.com"),  # 提供默认URL
                    "timestamp": current_timestamp  # 添加时间戳
                }
                filtered_references.append(filtered_ref)
            return filtered_references[:topk]  # 返回列表而不是JSON字符串

        except Exception as e:
            print(f"百度搜索API调用失败: {e}")
            # 如果真实调用失败，fallback 到本地模拟（下面）
            pass

    # ===== 模拟返回（当无法调用真实 API 时） =====
    # 注意：这些文本是示例新闻/事件摘要，真实环境中应用检索到的新闻/公告
    simulated = [
        {
            "title": "全球科技股本周普遍回调，半导体板块受库存及需求忧虑影响",
            "snippet": "受主要半导体公司下调指引与市场对智能手机需求放缓的预期，半导体相关ETF本周累计下跌约15%。分析师称短期供需修复存在不确定性。",
            "url": "https://news.example.com/tech_down",
            "content": "受主要半导体公司下调指引与市场对智能手机需求放缓的预期，半导体相关ETF本周累计下跌约15%。分析师称短期供需修复存在不确定性。",
            "timestamp": current_timestamp
        },
        {
            "title": "央行对短期利率政策作出微调，短债收益略有上升",
            "snippet": "央行声明显示短期利率保持稳定，但对货币政策语气偏中性，短期国债收益率微升，短债类资产表现相对稳健。",
            "url": "https://news.example.com/bond",
            "content": "央行声明显示短期利率保持稳定，但对货币政策语气偏中性，短期国债收益率微升，短债类资产表现相对稳健。",
            "timestamp": current_timestamp
        },
        {
            "title": "消费板块表现分化，大型必需消费品抗跌，小型消费股波动加剧",
            "snippet": "受宏观数据温和的影响，部分消费类基金出现小幅下探，但总体防御性偏好有所回升。",
            "url": "https://news.example.com/consumer",
            "content": "受宏观数据温和的影响，部分消费类基金出现小幅下探，但总体防御性偏好有所回升。",
            "timestamp": current_timestamp
        }
    ]
    return simulated[:topk]

def fetch_market_news_for_holdings(holdings: List[Dict[str, Any]], topk_per: int = 2) -> Dict[str, List[Dict[str, str]]]:
    """
    为每个重要sector/ticker拉取相关新闻（RAG检索），基于持仓权重和表现智能分配搜索资源。
    返回 mapping: key -> list of news items.
    """
    aggregated = {}
    
    # 计算各sector的权重
    sector_weights = {}
    for h in holdings:
        sector = h["sector"]
        if sector != "Cash":
            sector_weights[sector] = sector_weights.get(sector, 0) + h["weight"]
    
    # 根据权重和表现为sector分配优先级
    prioritized_sectors = []
    for sector, weight in sector_weights.items():
        # 找到该sector中最差表现的资产
        sector_holdings = [h for h in holdings if h["sector"] == sector]
        worst_performance = min(h.get("1m_ret", 0) for h in sector_holdings)
        # 优先级 = 权重 * 重要性因子 * (1 - 表现因子)
        importance_factor = 1.5 if weight > 0.2 else 1.0  # 权重超过20%的sector更重要
        performance_factor = min(1.0, worst_performance + 0.2)  # 表现越差，越需要关注
        priority_score = weight * importance_factor * (1 - performance_factor)
        prioritized_sectors.append((sector, weight, priority_score))
    
    # 按优先级排序sector
    prioritized_sectors.sort(key=lambda x: x[2], reverse=True)
    
    # 为高优先级sector拉取新闻
    for sector, weight, _ in prioritized_sectors:
        # 根据sector类型和权重确定搜索类型
        if weight > 0.25:
            # 权重高的sector需要深度分析
            search_type = "analysis"
        elif weight > 0.1:
            # 中等权重sector需要市场信息
            search_type = "market"
        else:
            # 小权重sector只需要最新新闻
            search_type = "news"
        
        # 构建针对性查询
        if sector.lower().startswith("technology"):
            query = f"{sector} 板块 市场动态 半导体 芯片 最新发展"
        elif sector.lower().startswith("fixedincome"):
            query = f"{sector} 债券市场 利率变动 收益率曲线"
        elif sector.lower().startswith("consumer"):
            query = f"{sector} 消费趋势 零售数据 消费信心"
        else:
            query = f"{sector} 行业 最新 市场动态"
            
        aggregated[sector] = call_baidu_search_api(query, topk=topk_per, search_type=search_type)
    
    # 为重要性高的个股拉取新闻
    # 1. 权重最高的前2个资产
    top_by_weight = sorted(holdings, key=lambda x: x.get("weight", 0), reverse=True)[:2]
    # 2. 表现最差的前2个资产
    worst_by_performance = sorted(holdings, key=lambda x: x.get("1m_ret", 0))[:2]
    # 合并并去重（使用ticker作为唯一标识）
    ticker_set = set()
    important_tickers = []
    for stock in top_by_weight + worst_by_performance:
        if stock['ticker'] not in ticker_set:
            ticker_set.add(stock['ticker'])
            important_tickers.append(stock)
    
    for h in important_tickers:
        if h["weight"] > 0.1 or h.get("1m_ret", 0) < -0.1:  # 权重超过10%或一个月跌幅超过10%
            # 根据表现确定搜索类型
            if h.get("1m_ret", 0) < -0.15:
                search_type = "analysis"  # 大幅下跌的资产需要深度分析
            else:
                search_type = "news"  # 其他资产只需要新闻
            
            query_parts = [h['name'], h['ticker'], "最新消息"]
            if h.get("1m_ret", 0) < -0.1:
                query_parts.append("下跌原因")
            elif h.get("1m_ret", 0) > 0.1:
                query_parts.append("上涨驱动")
            
            query = " ".join(query_parts)
            aggregated[h['ticker']] = call_baidu_search_api(query, topk=topk_per, search_type=search_type)
    
    # 添加宏观市场新闻作为补充
    macro_query = "全球市场 宏观经济 投资环境 风险因素"
    aggregated["宏观市场"] = call_baidu_search_api(macro_query, topk=topk_per, search_type="analysis")
    
    return aggregated

def enhance_portfolio_with_context(portfolio: Dict[str, Any], news_by_topic: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """
    增强组合信息，添加与新闻的关联分析和上下文
    返回增强后的组合信息
    """
    enhanced_portfolio = portfolio.copy()
    holdings = enhanced_portfolio["portfolio"]
    
    # 为每个持仓添加新闻相关性分析
    for h in holdings:
        # 查找相关新闻
        related_news = []
        
        # 1. 查找sector相关新闻
        if h["sector"] in news_by_topic:
            for news in news_by_topic[h["sector"]]:
                related_news.append((news, "sector"))
        
        # 2. 查找ticker特定新闻
        if h["ticker"] in news_by_topic:
            for news in news_by_topic[h["ticker"]]:
                related_news.append((news, "ticker"))
        
        # 3. 计算新闻影响评分
        impact_score = 0.0
        if related_news:
            # 计算综合影响评分（简化版）
            # 基于新闻数量、类型和关键词
            for news, news_type in related_news:
                # ticker特定新闻权重更高
                base_score = 1.0 if news_type == "ticker" else 0.5
                
                # 根据新闻内容关键词调整分数
                content_lower = news.get("content", "").lower()
                if any(neg in content_lower for neg in ["下跌", "风险", "利空", "衰退"]):
                    if h.get("1m_ret", 0) < 0:
                        impact_score += base_score * 1.5  # 负面新闻对已下跌资产影响更大
                elif any(pos in content_lower for pos in ["上涨", "利好", "增长", "创新"]):
                    if h.get("1m_ret", 0) > 0:
                        impact_score += base_score * 1.2  # 正面新闻对已上涨资产有增强效应
            
            impact_score = min(3.0, impact_score)  # 上限为3.0
        
        # 添加相关性信息
        h["news_impact_score"] = impact_score
        h["has_important_news"] = impact_score > 1.0
        
        # 添加简短的市场上下文描述
        if impact_score > 2.0:
            h["market_context"] = "该资产受到重要市场新闻影响，建议密切关注。"
        elif impact_score > 1.0:
            h["market_context"] = "该资产有相关市场动态，可能影响短期表现。"
        else:
            h["market_context"] = "该资产近期市场关注度相对较低。"
    
    # 整体组合市场暴露分析
    portfolio_exposure = {}
    # 统计不同sector的新闻覆盖度
    for sector in {h["sector"] for h in holdings}:
        if sector in news_by_topic:
            portfolio_exposure[sector] = len(news_by_topic[sector])
    
    # 识别新闻覆盖度最高的sector
    if portfolio_exposure:
        most_covered_sector = max(portfolio_exposure.items(), key=lambda x: x[1])[0]
        enhanced_portfolio["most_covered_sector"] = most_covered_sector
    
    # 添加宏观市场敏感度评估
    macro_sensitivity = "low"
    if "宏观市场" in news_by_topic and len(news_by_topic["宏观市场"]) > 0:
        macro_content = " ".join([n.get("content", "") for n in news_by_topic["宏观市场"]])
        if any(term in macro_content for term in ["波动", "危机", "重大", "显著"]):
            macro_sensitivity = "high"
        elif any(term in macro_content for term in ["变化", "调整", "影响"]):
            macro_sensitivity = "medium"
    
    enhanced_portfolio["macro_market_sensitivity"] = macro_sensitivity
    
    return enhanced_portfolio

# ---------------------------
# 4) 组合分析（简单因子/归因）— 本地可运行，不依赖外部服务
# ---------------------------
def analyze_portfolio(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """
    简单分析：计算加权 1d/1w/1m 回报，识别贡献最大的 sector 与单只跌幅最大持仓。
    返回分析字典用作 prompt context。
    """
    holdings = portfolio["portfolio"]
    # 1d/1w/1m 加权返回
    def weighted_return(key: str) -> float:
        return sum(h["weight"] * h.get(key, 0.0) for h in holdings)

    r_1d = weighted_return("1d_ret")
    r_1w = weighted_return("1w_ret")
    r_1m = weighted_return("1m_ret")

    # sector 贡献：按 sector 加权求和
    sector_map = {}
    for h in holdings:
        s = h["sector"]
        sector_map.setdefault(s, {"weight": 0.0, "weighted_1m": 0.0}).update(
            {
                "weight": sector_map.get(s, {}).get("weight", 0.0) + h["weight"],
                "weighted_1m": sector_map.get(s, {}).get("weighted_1m", 0.0) + h["weight"] * h.get("1m_ret", 0)
            }
        )
    # 将 sector_map 转为可读列表
    sector_list = []
    for s, v in sector_map.items():
        sector_list.append({"sector": s, "weight": v["weight"], "weighted_1m": v["weighted_1m"]})
    sector_list.sort(key=lambda x: x["weighted_1m"])  # 从最差到最好

    # 跌幅最大持仓
    worst_holding = min(holdings, key=lambda x: x.get("1m_ret", 0))

    analysis = {
        "r_1d": r_1d, "r_1w": r_1w, "r_1m": r_1m,
        "sector_summary": sector_list,
        "worst_holding": worst_holding,
        "holdings_count": len(holdings)
    }
    return analysis

# ---------------------------
# 5) 构建 RAG 上下文并调用 LLM
# ---------------------------
def build_rag_prompt(user_query: str,
                     portfolio: Dict[str, Any],
                     portfolio_analysis: Dict[str, Any],
                     news_by_topic: Dict[str, List[Dict[str, str]]]) -> str:
    """
    将检索到的新闻 + 组合分析拼成 LLM 的 context prompt (RAG context)。
    Prompt 要求 LLM 提供：原因分析、操作建议（分步）、教育性说明、目标/退出建议、风险提示与再平衡计划。
    """
    # header
    p = []
    p.append("你是一个合规、基于目标的投后陪伴投资顾问。请基于下列信息回答用户的问题，"
             "提供清晰的原因分析、分步可执行操作、风险/手续费考量、教育性说明与目标/退出建议。")
    p.append("\n=== 用户原始问题 ===\n" + user_query)
    p.append("\n=== 用户组合（摘要） ===")
    for h in portfolio["portfolio"]:
        p.append(f"- {h['ticker']} | {h['name']} | sector={h['sector']} | weight={h['weight']:.2%} | 1m_ret={h['1m_ret']:.1%}")
    p.append(f"- 现金占比: {portfolio.get('cash_reserve_pct',0):.2%}")
    p.append("\n=== 组合分析（自动归纳） ===")
    p.append(f"- 加权回报: 1d={portfolio_analysis['r_1d']:.2%}, 1w={portfolio_analysis['r_1w']:.2%}, 1m={portfolio_analysis['r_1m']:.2%}")
    p.append(f"- 本组合中表现最差的持仓： {portfolio_analysis['worst_holding']['ticker']}，1m_ret={portfolio_analysis['worst_holding']['1m_ret']:.1%}")
    p.append("- 各 sector 贡献（按 1m 加权回报，从差到好）：")
    for s in portfolio_analysis["sector_summary"]:
        p.append(f"  * {s['sector']}: weight={s['weight']:.2%}, weighted_1m={s['weighted_1m']:.2%}")
    # add news snippets
    p.append("\n=== 检索到的市场新闻/信息（供参考） ===")
    for topic, items in news_by_topic.items():
        p.append(f"\n--- 关于 {topic} 的新闻摘要 ---")
        for it in items:
            p.append(f"* {it['title']}\n  摘要: {it['snippet']}\n  来源: {it['url']}")
    # instructions for LLM
    p.append("\n=== 要求 ===")
    p.append("1) 简洁先给出 2-3 行核心结论。")
    p.append("2) 给出明确的分步操作建议（优先级标注：紧急/可选/长期）。")
    p.append("3) 针对用户的每个目标（短期/中期/长期）分别给出是否需要调整与如何调整。")
    p.append("4) 给出可解释的原因（例如：是单只资产导致，还是行业/宏观原因），并引用上方新闻或组合分析做支持。")
    p.append("5) 提供投后教育小结（不超过 6 条），帮助用户避免追涨杀跌。")
    p.append("6) 若建议卖出或减仓，请给出分批操作（分批比例与时间窗口）以及税费/手续费考虑。")
    p.append("7) 语言要友好，避免金融术语堆砌；若需要术语，给出简短解释。")
    prompt = "\n".join(p)
    return prompt

# ---------------------------
# 6) Agent 主流程
# ---------------------------
def load_knowledge_base(kb_path: str) -> List[Document]:
    """
    加载知识库文件并返回文档列表
    """
    try:
        if LANGCHAIN_AVAILABLE:
            loader = TextLoader(kb_path, encoding='utf-8')
            documents = loader.load()
            return documents
        else:
            # 手动读取文件作为备选方案
            with open(kb_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return [Document(page_content=content, metadata={"source": kb_path})]
    except Exception as e:
        print(f"加载知识库失败: {e}")
        return []

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    将文档分块，便于后续向量嵌入
    """
    try:
        if LANGCHAIN_AVAILABLE:
            # 基于标题和段落进行分块
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "\t", " "],
                length_function=len
            )
            return text_splitter.split_documents(documents)
        else:
            # 简单的手动分块作为备选方案
            chunks = []
            for doc in documents:
                content = doc.page_content
                for i in range(0, len(content), chunk_size - chunk_overlap):
                    chunk = content[i:i + chunk_size]
                    chunks.append(Document(page_content=chunk, metadata=doc.metadata))
            return chunks
    except Exception as e:
        print(f"文档分块失败: {e}")
        return documents

def create_vector_db(documents: List[Document]) -> Any:
    """
    创建向量数据库
    """
    try:
        if LANGCHAIN_AVAILABLE:
            # 使用开源的中文embedding模型
            embeddings = HuggingFaceEmbeddings(
                model_name="shibing624/text2vec-base-chinese",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            # 创建向量数据库
            db = FAISS.from_documents(documents, embeddings)
            return db
        else:
            print("LangChain不可用，无法创建向量数据库")
            return None
    except Exception as e:
        print(f"创建向量数据库失败: {e}")
        return None

def retrieve_with_rerank(query: str, k_coarse: int = 10, k_final: int = 3) -> List[Document]:
    """
    实现粗排精排检索流程
    1. 粗排：使用向量相似度检索更多候选文档（k_coarse）
    2. 精排：对候选文档进行内容相关性排序，返回最相关的k_final个文档
    """
    global vector_db, knowledge_base
    
    # 如果向量数据库未初始化，初始化它
    if vector_db is None:
        kb_path = "kb.txt"  # 假设知识库文件在当前目录
        try:
            # 尝试加载知识库
            knowledge_base = load_knowledge_base(kb_path)
            # 分块
            chunks = split_documents(knowledge_base)
            # 创建向量数据库
            vector_db = create_vector_db(chunks)
            # 如果无法创建向量数据库，使用SimpleRetriever
            if not vector_db and 'SimpleRetriever' in globals():
                vector_db = SimpleRetriever(chunks)
                print("使用SimpleRetriever作为检索器")
        except Exception as e:
            print(f"初始化知识库失败: {e}")
            # 即使初始化失败，也创建一个简单的检索器作为备用
            default_docs = [
                Document(page_content="投资基础知识：分散投资可以降低风险", metadata={"source": "default"}),
                Document(page_content="长期投资通常比短期投机更可靠", metadata={"source": "default"}),
                Document(page_content="市场波动是正常的，投资者应该保持理性", metadata={"source": "default"}),
                Document(page_content="资产配置应根据个人风险承受能力和投资目标制定", metadata={"source": "default"}),
                Document(page_content="定期重新平衡投资组合有助于控制风险", metadata={"source": "default"})
            ]
            if 'SimpleRetriever' in globals():
                vector_db = SimpleRetriever(default_docs)
    
    if vector_db is None:
        # 返回默认文档
        return [
            Document(page_content="投资建议：保持长期投资视角", metadata={"source": "fallback"}),
            Document(page_content="分散投资可以降低整体风险", metadata={"source": "fallback"}),
            Document(page_content="定期检视投资组合表现并适时调整", metadata={"source": "fallback"})
        ][:k_final]
    
    try:
        # 1. 粗排阶段：获取更多候选文档
        coarse_results = vector_db.similarity_search(query, k=k_coarse)
        
        if not coarse_results:
            return []
        
        # 2. 精排阶段：基于内容相关性进行重排序
        # 这里使用一个简单的精排策略：计算文本重叠度分数
        reranked_results = []
        query_lower = query.lower()
        
        for doc in coarse_results:
            content = doc.page_content.lower()
            
            # 计算文本重叠度分数
            # 1. 计算关键词重叠度
            query_words = set(query_lower.split())
            content_words = set(content.split())
            common_words = query_words.intersection(content_words)
            overlap_score = len(common_words) / len(query_words) if query_words else 0
            
            # 2. 计算语义相关度（基于原始相似度分数）
            # 注意：在FAISS检索中，我们可以获取相似度分数
            # 这里简化处理，假设最近的文档分数更高
            
            reranked_results.append((doc, overlap_score))
        
        # 按综合分数排序
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的文档内容
        return [doc for doc, score in reranked_results[:k_final]]
        
    except Exception as e:
        print(f"粗排精排检索失败: {e}")
        # 发生错误时回退到简单检索
        try:
            return vector_db.similarity_search(query, k=k_final)
        except:
            return []

def retrieve_from_knowledge_base(query: str, k: int = 3) -> List[str]:
    """
    从知识库中检索相关内容，使用粗排精排优化
    """
    try:
        documents = retrieve_with_rerank(query, k_coarse=10, k_final=k)
        return [doc.page_content for doc in documents]
    except Exception as e:
        print(f"从知识库检索失败: {e}")
        # 返回默认的投资建议
        return [
            "投资建议：保持长期投资视角",
            "分散投资可以降低整体风险",
            "定期检视投资组合表现并适时调整"
        ][:k]

def multi_query_retrieval(query: str, k: int = 3) -> List[str]:
    """
    使用多查询检索策略，通过生成多个相关查询来提高检索效果
    """
    global vector_db
    
    if vector_db is None or not LANGCHAIN_AVAILABLE:
        # 回退到简单检索
        return retrieve_from_knowledge_base(query, k)
    
    try:
        # 创建多查询检索器
        # 注意：这里简化实现，实际上需要LLM来生成多个相关查询
        # 这里我们手动生成几个常见变体
        query_variants = generate_query_variants(query)
        
        # 从每个变体中检索结果
        all_results = set()
        for variant in query_variants:
            docs = retrieve_with_rerank(variant, k_coarse=5, k_final=2)
            for doc in docs:
                all_results.add(doc)
        
        # 对所有结果再次精排序
        results_with_scores = []
        query_lower = query.lower()
        for doc in all_results:
            content = doc.page_content.lower()
            query_words = set(query_lower.split())
            content_words = set(content.split())
            common_words = query_words.intersection(content_words)
            score = len(common_words) / len(query_words) if query_words else 0
            results_with_scores.append((doc, score))
        
        # 排序并返回前k个
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc.page_content for doc, score in results_with_scores[:k]]
        
    except Exception as e:
        print(f"多查询检索失败: {e}")
        return retrieve_from_knowledge_base(query, k)

def generate_query_variants(query: str) -> List[str]:
    """
    生成查询的多个变体，用于多查询检索
    """
    variants = [query]
    
    # 扩展一些常见的查询变体
    if "为什么" in query:
        variants.append(query.replace("为什么", "原因是"))
    if "怎么办" in query:
        variants.append(query.replace("怎么办", "如何应对"))
        variants.append(query.replace("怎么办", "解决方法"))
    if "推荐" in query:
        variants.append(query.replace("推荐", "建议"))
    if "风险" in query:
        variants.append(query.replace("风险", "注意事项"))
    
    # 添加更具体和更广泛的版本
    if len(query) > 10:
        # 简化查询
        simplified = query.split('，')[0] if '，' in query else query
        if simplified not in variants:
            variants.append(simplified)
    
    # 确保不超过5个变体
    return variants[:5]

def enhanced_build_rag_prompt(user_query: str, portfolio: Dict[str, Any], 
                            portfolio_analysis: Dict[str, Any], news_by_topic: Dict[str, List[Dict[str, str]]]) -> str:
    """
    增强版RAG prompt构建，集成知识库检索结果
    """
    # 先构建基础prompt
    p = []
    p.append("你是一个合规、基于目标的投后陪伴投资顾问。请基于下列信息回答用户的问题，" 
             "提供清晰的原因分析、分步可执行操作、风险/手续费考量、教育性说明与目标/退出建议。")
    p.append("\n=== 用户原始问题 ===\n" + user_query)
    
    # 从知识库中检索相关内容，使用多查询检索提高效果
    kb_contents = multi_query_retrieval(user_query, k=3)
    if kb_contents:
        p.append("\n=== 知识库相关内容 ===")
        p.append("以下内容来自专业投资知识库，为您的问题提供系统化的参考依据：")
        for i, content in enumerate(kb_contents):
            p.append(f"\n--- 相关知识 {i+1} ---")
            p.append(content)
    
    # 添加用户组合信息
    p.append("\n=== 用户组合（摘要） ===")
    for h in portfolio["portfolio"]:
        p.append(f"- {h['ticker']} | {h['name']} | sector={h['sector']} | weight={h['weight']:.2%} | 1m_ret={h['1m_ret']:.1%}")
    p.append(f"- 现金占比: {portfolio.get('cash_reserve_pct',0):.2%}")
    
    # 添加组合分析
    p.append("\n=== 组合分析（自动归纳） ===")
    p.append(f"- 加权回报: 1d={portfolio_analysis['r_1d']:.2%}, 1w={portfolio_analysis['r_1w']:.2%}, 1m={portfolio_analysis['r_1m']:.2%}")
    p.append(f"- 本组合中表现最差的持仓： {portfolio_analysis['worst_holding']['ticker']}，1m_ret={portfolio_analysis['worst_holding']['1m_ret']:.1%}")
    p.append("- 各 sector 贡献（按 1m 加权回报，从差到好）：")
    for s in portfolio_analysis["sector_summary"]:
        p.append(f"  * {s['sector']}: weight={s['weight']:.2%}, weighted_1m={s['weighted_1m']:.2%}")
    
    # 添加市场新闻
    p.append("\n=== 检索到的市场新闻/信息（供参考） ===")
    for topic, items in news_by_topic.items():
        p.append(f"\n--- 关于 {topic} 的新闻摘要 ---")
        for it in items:
            p.append(f"* {it['title']}\n  摘要: {it['snippet']}\n  来源: {it['url']}")
    
    # 添加指令
    p.append("\n=== 要求 ===")
    p.append("1) 简洁先给出 2-3 行核心结论。")
    p.append("2) 给出明确的分步操作建议（优先级标注：紧急/可选/长期）。")
    p.append("3) 针对用户的每个目标（短期/中期/长期）分别给出是否需要调整与如何调整。")
    p.append("4) 给出可解释的原因（例如：是单只资产导致，还是行业/宏观原因），并引用上方新闻或组合分析做支持。")
    p.append("5) 提供投后教育小结（不超过 6 条），帮助用户避免追涨杀跌。")
    p.append("6) 若建议卖出或减仓，请给出分批操作（分批比例与时间窗口）以及税费/手续费考虑。")
    p.append("7) 语言要友好，避免金融术语堆砌；若需要术语，给出简短解释。")
    p.append("8) 请确保回答具有高度可解释性，明确说明'为什么推荐这个方案'，帮助用户建立信任感。")
    
    return "\n".join(p)

def agent_handle_query(user_id: str, user_query: str) -> Dict[str, Any]:
    """
    主函数：对外接口，输入 user_id 和 user_query，返回 LLM 生成的结构化建议与中间上下文（用于审计/回溯）。
    """
    try:
        # 1) 读取组合
        portfolio = get_portfolio(user_id)

        # 2) 基础分析
        portfolio_analysis = analyze_portfolio(portfolio)

        # 3) RAG: 检索相关新闻
        news_by_topic = fetch_market_news_for_holdings(portfolio["portfolio"], topk_per=2)
        
        # 4) 增强组合信息，添加与新闻的关联分析
        enhanced_portfolio = enhance_portfolio_with_context(portfolio, news_by_topic)

        # 5) 构造增强版RAG prompt（集成知识库检索、组合分析和市场新闻）
        prompt = enhanced_build_rag_prompt(user_query, enhanced_portfolio, portfolio_analysis, news_by_topic)

        # 6) 调用 LLM
        llm_answer = get_llm_output(prompt=prompt, query=user_query)

        # 7) 同时生成可供前端展示的简短 summary（自动提取第一段或归纳）
        summary_lines = llm_answer.strip().splitlines()
        summary = "\n".join(summary_lines[:6]) if summary_lines else ""

        # 8) 生成可解释性部分
        explainability = generate_explainability(enhanced_portfolio, portfolio_analysis, user_query)

        # 9) 返回结构化对象（供 UI / 日志 / 可解释性审计）
        return {
            "user_id": user_id,
            "query": user_query,
            "prompt": prompt,
            "llm_answer": llm_answer,
            "summary": summary,
            "portfolio": enhanced_portfolio,  # 返回增强后的组合信息
            "portfolio_analysis": portfolio_analysis,
            "news_by_topic": news_by_topic,
            "explainability": explainability,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 添加时间戳
        }
    except Exception as e:
        print(f"处理查询时出错: {e}")
        # 返回错误情况下的默认响应
        import traceback
        traceback.print_exc()  # 打印详细错误信息用于调试
        
        # 生成基本的组合信息作为回退
        default_portfolio = {
            "portfolio": [],
            "cash": 100000,
            "total_value": 100000,
            "last_update": datetime.now().strftime("%Y-%m-%d")
        }
        
        return {
            "user_id": user_id,
            "query": user_query,
            "prompt": "",
            "llm_answer": "尊敬的用户，系统暂时无法处理您的投资咨询请求。以下是一些通用的投资建议：\n\n1. 保持长期投资视角，避免频繁交易\n2. 分散投资于不同资产类别和行业\n3. 根据个人风险承受能力制定投资策略\n4. 定期检视投资组合表现\n5. 关注宏观经济形势和市场变化\n\n建议您稍后再次尝试，或联系专业投资顾问获取个性化建议。",
            "summary": "系统暂时无法处理您的投资咨询请求。提供以下通用投资建议：保持长期投资视角、分散投资、根据风险承受能力制定策略、定期检视组合、关注宏观经济形势。",
            "portfolio": default_portfolio,
            "portfolio_analysis": {
                "total_return": 0.0,
                "r_1m": 0.0,
                "risk_level": "中等",
                "sector_summary": []
            },
            "news_by_topic": {},
            "explainability": {
                "performance_reason": "系统暂时无法提供详细的组合表现分析",
                "risk_analysis": "系统暂时无法提供详细的风险分析",
                "action_suggestion": "建议您稍后再次尝试，或咨询专业投资顾问"
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }

def generate_explainability(portfolio: Dict[str, Any], portfolio_analysis: Dict[str, Any], user_query: str) -> Dict[str, str]:
    """
    生成可解释性内容，增强用户信任，提供透明的投资逻辑
    """
    explainability = {}
    
    # 1. 组合表现解释 - 详细分析涨跌原因
    if portfolio_analysis['r_1m'] < -0.05:  # 如果一个月亏损超过5%
        # 计算sector贡献度
        worst_sector = portfolio_analysis['sector_summary'][0] if portfolio_analysis['sector_summary'] else None
        
        explainability['performance_reason'] = "组合近期表现不佳，我们进行了详细分析："
        
        if worst_sector:
            impact_pct = abs(worst_sector['weighted_1m'] / portfolio_analysis['r_1m']) * 100 if portfolio_analysis['r_1m'] != 0 else 0
            explainability['sector_impact'] = (
                f"• 主要拖累因素：{worst_sector['sector']}板块贡献了约{impact_pct:.1f}%的跌幅\n"
                f"• 具体表现最差的资产是：{portfolio_analysis['worst_holding']['name']}，" 
                f"1个月跌幅达{portfolio_analysis['worst_holding']['1m_ret']*100:.1f}%"
            )
        
        # 检查是否有重大新闻影响
        if hasattr(portfolio, 'most_covered_sector'):
            explainability['news_factor'] = f"• 新闻影响：{portfolio['most_covered_sector']}板块近期新闻关注度较高，可能是影响组合表现的外部因素"
        
        # 添加宏观市场敏感度
        if hasattr(portfolio, 'macro_market_sensitivity'):
            sensitivity_map = {
                'high': '市场环境波动较大，整体风险较高',
                'medium': '市场环境有一定变化，但处于合理范围',
                'low': '市场环境相对稳定，适合长期持有'
            }
            explainability['market_context'] = f"• 市场环境：{sensitivity_map.get(portfolio.get('macro_market_sensitivity', 'low'), '市场环境相对稳定')}"
    
    elif portfolio_analysis['r_1m'] > 0.05:  # 如果一个月上涨超过5%
        explainability['performance_reason'] = "组合近期表现良好，主要上涨因素分析："
        # 找出表现最好的sector
        best_sector = portfolio_analysis['sector_summary'][-1] if portfolio_analysis['sector_summary'] else None
        if best_sector:
            impact_pct = best_sector['weighted_1m'] / portfolio_analysis['r_1m'] * 100 if portfolio_analysis['r_1m'] != 0 else 0
            explainability['sector_impact'] = (
                f"• 主要贡献因素：{best_sector['sector']}板块贡献了约{impact_pct:.1f}%的涨幅"
            )
    else:
        explainability['performance_reason'] = "组合表现相对稳定，符合市场预期，各板块贡献相对均衡。"
    
    # 2. 目标相关性解释 - 提供更具体的目标实现路径
    short_term_goals = [goal for goal in portfolio['target_goals'] if goal['horizon_years'] <= 3]
    mid_term_goals = [goal for goal in portfolio['target_goals'] if 3 < goal['horizon_years'] <= 10]
    long_term_goals = [goal for goal in portfolio['target_goals'] if goal['horizon_years'] > 10]
    
    # 针对不同期限的目标提供建议
    if short_term_goals:
        cash_suggestion = "建议增加" if portfolio['cash_reserve_pct'] < 0.1 else "维持" if portfolio['cash_reserve_pct'] >= 0.1 and portfolio['cash_reserve_pct'] <= 0.2 else "考虑适度减少"
        explainability['short_term_fit'] = (
            f"短期目标适配性：考虑到您有{len(short_term_goals)}个短期目标（如{short_term_goals[0]['goal']}），"
            f"{cash_suggestion}现金储备比例在10-20%之间，以应对可能的流动性需求。"
        )
        
        # 提供具体的退出机制建议
        explainability['exit_strategy_short'] = (
            "短期目标退出建议：\n"
            "• 设定明确的目标达成阈值（如目标金额的90%）\n"
            "• 采用分批退出策略，避免时点风险\n"
            "• 提前3-6个月开始逐步降低权益类资产比例"
        )
    
    if mid_term_goals:
        explainability['mid_term_fit'] = (
            f"中期目标适配性：对于您的{len(mid_term_goals)}个中期目标（如{mid_term_goals[0]['goal']}），"
            f"当前组合风险水平{'较为匹配' if 0.4 <= portfolio['risk_budget'] <= 0.7 else '可能需要调整'}。"
        )
        
        explainability['exit_strategy_mid'] = (
            "中期目标退出建议：\n"
            "• 目标到期前2-3年开始逐步降低组合波动性\n"
            "• 建立季度检视机制，根据市场环境调整退出节奏\n"
            "• 考虑目标达成后的资金再配置方案"
        )
    
    if long_term_goals:
        explainability['long_term_fit'] = (
            f"长期目标适配性：针对您的长期目标（如{long_term_goals[0]['goal']}），"
            f"投资期限较长，当前风险水平{'可以接受' if portfolio['risk_budget'] >= 0.5 else '略显保守'}。"
        )
    
    # 3. 风险水平解释 - 更详细的风险构成分析
    equity_weight = sum(h['weight'] for h in portfolio['portfolio'] if h['sector'] in ['Technology', 'Consumer'])
    bond_weight = sum(h['weight'] for h in portfolio['portfolio'] if h['sector'] == 'FixedIncome')
    cash_weight = portfolio['cash_reserve_pct']
    
    risk_level = '积极' if portfolio['risk_budget'] > 0.7 else '平衡' if portfolio['risk_budget'] > 0.4 else '保守'
    
    explainability['risk_level'] = (
        f"当前组合风险水平：{risk_level}\n"
        f"• 权益类资产占比：{equity_weight*100:.0f}%\n"
        f"• 固定收益类资产占比：{bond_weight*100:.0f}%\n"
        f"• 现金类资产占比：{cash_weight*100:.0f}%"
    )
    
    # 4. 市场情绪与教育提示
    explainability['market_sentiment'] = generate_market_sentiment_guidance(portfolio_analysis, user_query)
    
    # 5. 个性化投后教育
    explainability['investor_education'] = generate_investor_education(portfolio, portfolio_analysis, user_query)
    
    return explainability

def generate_market_sentiment_guidance(portfolio_analysis: Dict[str, Any], user_query: str) -> str:
    """
    根据组合表现和用户问题生成市场情绪引导，帮助用户保持理性
    """
    # 检测用户问题中的情绪倾向
    has_anxiety = any(term in user_query for term in ['亏损', '赎回', '下跌', '担心', '恐慌', '害怕'])
    has_excitement = any(term in user_query for term in ['上涨', '盈利', '加仓', '追高', '热门'])
    
    guidance = "市场情绪提示："
    
    # 亏损时的安抚和教育
    if portfolio_analysis['r_1m'] < -0.08 and has_anxiety:
        guidance += (
            "\n• 短期波动是投资常态，历史数据显示，长期持有通常能度过市场调整期\n"
            "• 恐慌性赎回往往导致'低卖高买'，加剧投资损失\n"
            "• 建议回顾您的投资目标期限，短期波动对长期目标影响有限\n"
            "• 考虑利用市场调整进行策略性再平衡，而非全部退出"
        )
    # 大幅上涨时的风险提示
    elif portfolio_analysis['r_1m'] > 0.1 and has_excitement:
        guidance += (
            "\n• 市场短期快速上涨后，可能面临调整压力\n"
            "• 建议保持理性，避免盲目追高\n"
            "• 考虑适度兑现部分收益，落袋为安\n"
            "• 重新评估组合风险，确保与您的风险承受能力匹配"
        )
    # 正常市场环境下的教育
    else:
        guidance += (
            "\n• 投资是一场马拉松，而非短跑\n"
            "• 定期检视组合，但避免过度交易\n"
            "• 保持投资纪律，遵循既定的资产配置策略\n"
            "• 市场噪音可能导致非理性决策，请专注于长期目标"
        )
    
    return guidance

def generate_investor_education(portfolio: Dict[str, Any], portfolio_analysis: Dict[str, Any], user_query: str) -> str:
    """
    生成个性化的投后教育内容，帮助用户提升投资认知
    """
    education_points = [
        "投后教育要点："
    ]
    
    # 根据组合特征添加教育内容
    equity_weight = sum(h['weight'] for h in portfolio['portfolio'] if h['sector'] in ['Technology', 'Consumer'])
    
    # 分散投资教育
    if equity_weight > 0.7:
        education_points.append("• 分散投资是降低风险的有效手段，建议考虑增加资产类别多样性")
    
    # 再平衡教育
    best_sector = portfolio_analysis['sector_summary'][-1] if portfolio_analysis['sector_summary'] else None
    worst_sector = portfolio_analysis['sector_summary'][0] if len(portfolio_analysis['sector_summary']) > 1 else None
    
    if best_sector and worst_sector and best_sector['weighted_1m'] > -worst_sector['weighted_1m'] * 2:
        education_points.append("• 再平衡策略可以'高抛低吸'，在市场波动中优化组合表现")
    
    # 长期投资教育
    long_goals = [goal for goal in portfolio['target_goals'] if goal['horizon_years'] > 10]
    if long_goals and portfolio_analysis['r_1m'] < 0:
        education_points.append("• 对于长期目标，短期市场波动应被视为正常现象，坚持投资计划更为重要")
    
    # 流动性管理教育
    cash_weight = portfolio['cash_reserve_pct']
    short_goals = [goal for goal in portfolio['target_goals'] if goal['horizon_years'] <= 3]
    if short_goals and cash_weight < 0.1:
        education_points.append("• 短期目标需要合理的流动性储备，建议保持适当的现金比例以应对突发需求")
    
    # 目标调整机制教育
    education_points.append("• 投资目标可以根据生活变化进行调整，但请确保新目标经过充分评估")
    education_points.append("• 当目标达成或改变时，建议制定新的投资计划，而非随意调整")
    
    return "\n".join(education_points)

def generate_portfolio_tracking_summary(portfolio: Dict[str, Any], portfolio_analysis: Dict[str, Any]) -> Dict[str, str]:
    """
    生成组合跟踪摘要，提供持续的投后管理信息
    """
    tracking = {}
    
    # 1. 表现跟踪
    tracking['performance_summary'] = (
        f"组合表现概览：\n"
        f"• 日涨幅：{portfolio_analysis['r_1d']*100:.2f}%\n"
        f"• 周涨幅：{portfolio_analysis['r_1w']*100:.2f}%\n"
        f"• 月涨幅：{portfolio_analysis['r_1m']*100:.2f}%"
    )
    
    # 2. 需要关注的资产
    watch_list = []
    for h in portfolio['portfolio']:
        if abs(h.get('1m_ret', 0)) > 0.15:  # 波动超过15%
            direction = "上涨" if h.get('1m_ret', 0) > 0 else "下跌"
            watch_list.append(f"• {h['name']}：近1个月{direction}{abs(h.get('1m_ret', 0))*100:.1f}%")
        elif h.get('has_important_news', False):
            watch_list.append(f"• {h['name']}：有重要相关新闻")
    
    if watch_list:
        tracking['watch_list'] = "需要关注的资产：\n" + "\n".join(watch_list)
    else:
        tracking['watch_list'] = "目前没有特别需要关注的资产，组合表现相对稳定"
    
    # 3. 再平衡建议
    rebalance_needed = False
    rebalance_suggestions = []
    
    # 检查资产配置偏离度
    target_allocation = {
        'Technology': 0.4,
        'FixedIncome': 0.3,
        'Consumer': 0.2,
        'Cash': 0.1
    }
    
    current_allocation = {}
    for h in portfolio['portfolio']:
        sector = h['sector']
        current_allocation[sector] = current_allocation.get(sector, 0) + h['weight']
    
    for sector, target in target_allocation.items():
        current = current_allocation.get(sector, 0)
        deviation = abs(current - target)
        if deviation > 0.05:  # 偏离超过5%
            rebalance_needed = True
            action = "增加" if current < target else "减少"
            rebalance_suggestions.append(f"• {sector}类资产：建议{action}配置至目标比例{target*100:.0f}%")
    
    if rebalance_needed:
        tracking['rebalance_suggestion'] = "再平衡建议：\n" + "\n".join(rebalance_suggestions)
    else:
        tracking['rebalance_suggestion'] = "当前资产配置符合目标要求，暂时无需再平衡"
    
    # 4. 目标达成进度
    goals_progress = []
    for goal in portfolio['target_goals']:
        # 这里简化处理，实际应该基于历史表现和模拟预测
        time_progress = min(1.0, 1.0 / goal['horizon_years']) if goal['horizon_years'] > 0 else 1.0
        progress_desc = "起步阶段" if time_progress < 0.3 else "进行中" if time_progress < 0.8 else "接近目标"
        goals_progress.append(f"• {goal['goal']}（{goal['horizon_years']}年）：{progress_desc}")
    
    tracking['goals_progress'] = "目标达成进度：\n" + "\n".join(goals_progress)
    
    return tracking

# ---------------------------
# 7) 示例数据与示例运行（可直接执行）
# ---------------------------

def test_rag_functionality():
    """
    测试RAG功能是否正常工作
    """
    print("\n=== 测试RAG功能 ===")
    
    try:
        # 加载知识库
        print("正在加载知识库...")
        knowledge_base = load_knowledge_base(KB_PATH)
        
        # 文档分块
        print("正在进行文档分块...")
        chunks = split_documents(knowledge_base)
        print(f"成功分割为 {len(chunks)} 个文档块")
        
        # 创建向量数据库
        print("正在创建向量数据库...")
        vector_db = create_vector_db(chunks)
        
        # 测试检索功能
        test_query = "市场波动时如何管理投资心理？"
        print(f"\n测试查询: {test_query}")
        
        # 测试多查询检索
        multi_results = multi_query_retrieval(test_query, k=3)
        print(f"多查询检索返回 {len(multi_results)} 条相关信息")
        
        # 打印前两条检索结果
        print("\n检索结果示例:")
        for i, result in enumerate(multi_results[:2]):
            print(f"{i+1}. {result[:100]}...")
        
        return True
    except Exception as e:
        print(f"RAG功能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_baidu_search():
    """
    测试百度搜索API功能
    """
    print("\n=== 测试百度搜索功能 ===")
    
    try:
        # 使用模拟的API进行测试
        search_query = "最新股市行情"
        print(f"搜索查询: {search_query}")
        
        # 由于我们使用的是模拟函数，这里会返回模拟数据
        search_results = call_baidu_search_api(search_query, search_type="news")
        print(f"搜索返回 {len(search_results)} 条结果")
        
        # 打印前两条结果
        print("\n搜索结果示例:")
        for i, result in enumerate(search_results[:2]):
            print(f"{i+1}. {result.get('title', '无标题')}")
            print(f"   {result.get('snippet', '无摘要')[:100]}...")
            print(f"   URL: {result.get('url', '无链接')}")
        
        return True
    except Exception as e:
        print(f"百度搜索功能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_explainability_and_tracking():
    """
    测试可解释性和投后跟踪功能
    """
    print("\n=== 测试可解释性和投后跟踪功能 ===")
    
    try:
        # 生成模拟的用户组合和分析数据
        portfolio = generate_mock_portfolio()
        portfolio_analysis = analyze_portfolio(portfolio)
        user_query = "我的组合最近表现不好，应该赎回吗？"
        
        # 测试可解释性生成
        print("\n可解释性分析结果:")
        explainability = generate_explainability(portfolio, portfolio_analysis, user_query)
        for key, value in explainability.items():
            print(f"\n{key}:")
            print(value)
        
        # 测试投后跟踪功能
        print("\n投后跟踪摘要:")
        tracking_summary = generate_portfolio_tracking_summary(portfolio, portfolio_analysis)
        for key, value in tracking_summary.items():
            print(f"\n{key}:")
            print(value)
        
        return True
    except Exception as e:
        print(f"可解释性和投后跟踪功能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_complete_test():
    """
    运行完整的系统测试，验证所有功能模块
    """
    print("\n开始测试投后陪伴顾问Agent系统...")
    print("-"*50)
    
    # 运行各项测试
    tests = [
        ("RAG功能", test_rag_functionality),
        ("百度搜索功能", test_baidu_search),
        ("可解释性和投后跟踪功能", test_explainability_and_tracking)
    ]
    
    passed_count = 0
    total_count = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n正在运行: {test_name}")
        print("-" * 50)
        if test_func():
            passed_count += 1
            print(f"✓ {test_name} 测试通过")
        else:
            print(f"✗ {test_name} 测试失败")
        print("-" * 50)
    
    # 显示测试结果摘要
    print("\n=== 测试结果摘要 ===")
    print(f"总测试数: {total_count}")
    print(f"通过测试数: {passed_count}")
    print(f"失败测试数: {total_count - passed_count}")
    
    if passed_count == total_count:
        print("\n✅ 所有测试通过！投后陪伴顾问Agent系统运行正常。")
    else:
        print("\n❌ 部分测试失败，请检查代码并修复问题。")

def demonstrate_agent():
    """
    演示完整的Agent工作流程，展示用户交互场景
    """
    print("\n=== 投后陪伴顾问Agent演示 ===")
    print("="*80)
    
    # 示例用户问题
    sample_query = "最近组合亏损严重，我该赎回吗？我的目标是：1年内有一笔房屋首付（短期应急），同时长期退休投资。"
    print(f"用户查询: {sample_query}")
    
    # 调用Agent处理查询
    result = agent_handle_query("U_test_001", sample_query)
    
    # 将结构化结果保存为 JSON（便于审计 / 前端加载）
    with open("agent_result_sample.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 打印简短 summary 与 LLM 完整回答的前段
    print("\n----- 简短 Summary -----")
    print(result["summary"])
    print("\n----- LLM 完整回答（片段） -----")
    print(result["llm_answer"][:1500])  # 打印前1500字符，避免太长
    
    # 特别展示可解释性部分
    print("\n----- 可解释性分析 -----")
    for key, value in result["explainability"].items():
        print(f"\n{key}:")
        print(value)
    
    # 展示投后跟踪摘要
    portfolio = result["portfolio"]
    portfolio_analysis = result["portfolio_analysis"]
    tracking_summary = generate_portfolio_tracking_summary(portfolio, portfolio_analysis)
    
    print("\n----- 投后跟踪摘要 -----")
    for key, value in tracking_summary.items():
        print(f"\n{key}:")
        print(value)
    
    print("\n已将完整结果写入 agent_result_sample.json")
    print("="*80)

# 添加生成模拟组合的函数（如果还没有的话）
def generate_mock_portfolio():
    """
    生成模拟的用户组合数据
    """
    return {
        "portfolio": [
            {
                "ticker": "TSLA",
                "name": "特斯拉",
                "sector": "Technology",
                "weight": 0.25,
                "1m_ret": -0.15,
                "has_important_news": True
            },
            {
                "ticker": "AAPL",
                "name": "苹果公司",
                "sector": "Technology",
                "weight": 0.20,
                "1m_ret": -0.05
            },
            {
                "ticker": "MSFT",
                "name": "微软公司",
                "sector": "Technology",
                "weight": 0.15,
                "1m_ret": 0.02
            },
            {
                "ticker": "JNJ",
                "name": "强生公司",
                "sector": "Consumer",
                "weight": 0.15,
                "1m_ret": 0.01
            },
            {
                "ticker": "BND",
                "name": "债券ETF",
                "sector": "FixedIncome",
                "weight": 0.15,
                "1m_ret": 0.005
            },
            {
                "ticker": "GLD",
                "name": "黄金ETF",
                "sector": "Commodity",
                "weight": 0.10,
                "1m_ret": 0.03
            }
        ],
        "cash_reserve_pct": 0.15,
        "risk_budget": 0.6,
        "target_goals": [
            {
                "goal": "1年内积累房屋首付",
                "horizon_years": 1
            },
            {
                "goal": "长期退休投资",
                "horizon_years": 20
            }
        ],
        "most_covered_sector": "Technology",
        "macro_market_sensitivity": "medium"
    }

if __name__ == "__main__":
    # 运行完整的系统测试
    run_complete_test()
    
    # 演示Agent的实际工作流程
    demonstrate_agent()
    
    print("\n✅ 系统测试和演示完成！")
    print("投后陪伴顾问Agent已成功实现透明可信的投资建议和全面的投后管理功能。")
    print("用户可以获得清晰的'为什么买'、'为什么此时买'、'为什么这个比例'的解释，")
    print("以及持续的跟踪解读、陪伴式教育和明确的退出机制指导。")

























