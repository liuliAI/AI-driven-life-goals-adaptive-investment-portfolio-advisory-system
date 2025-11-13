# 智能投资组合推荐与投后管理系统

## 项目背景

在当前金融科技领域，个人投资者面临着诸多挑战，我们的系统致力于解决以下五大核心痛点：

### 痛点1：从"单品推荐"到"组合构建"的鸿沟
- **认知负荷高**：用户面对多个"优等生"基金时，陷入"选择悖论"，资金分配更是远超普通用户能力范围
- **责任转移**：平台推荐单个产品，但将组合构建风险完全转移给用户
- **组合构建门槛高**：推荐零散的单个金融产品，将复杂的资产配置任务抛给用户

### 痛点2：从"标签化"到"深度个性化"的缺失
- **数据维度单一**：目前个性化大多仅基于风险测评问卷，忽略消费习惯、持仓变化、人生阶段等宝贵数据
- **"风险偏好"不等于"风险需求"**：未能区分"为了不同目标而承担的不同风险"
- **个性化程度不足**：未能与用户具体的、可量化的人生目标紧密结合

### 痛点3：从"静态快照"到"动态画像"的滞后
- **用户是动态的**：年龄、收入、家庭状况、市场经验都在变化
- **市场是动态的**：经济周期、货币政策、行业轮动
- **组合是静态的**：推荐组合一旦构建，就与上述变化脱钩，导致"配置漂移"

### 痛点4：从"黑箱操作"到"透明可信"的信任缺失
- **推荐动机质疑**：用户怀疑推荐是否真的为自己好，还是因为平台能获得更高佣金
- **无法理解故无法信任**：不知道"为什么买A"、"为什么此时买"、"为什么这个比例"
- **透明逻辑缺失**：用户在市场波动时极易恐慌性赎回，透明的逻辑是帮助用户"拿得住"的关键

### 痛点5：投后跟踪管理的缺失与用户指导的不足
- **推荐后无人区**：推荐动作在用户购买完成后基本结束
- **缺乏持续跟踪与解读**：组合为什么涨/跌？是否需要操作？
- **缺乏陪伴式教育**：市场狂热时未提示风险，市场恐慌时未安抚情绪
- **退出机制不清晰**：达到目标后怎么办？目标改变后怎么办？

## 项目结构

```
szf_part/
├── data/                    # 数据文件夹
│   ├── profile.csv          # 用户画像数据
│   └── product.csv          # 金融产品数据
├── portfolio_recommender/   # 投资组合推荐模块（解决痛点1-3）
│   ├── main.py              # 主入口文件
│   ├── config.py            # 配置文件
│   ├── data_processing.py   # 数据预处理模块
│   ├── user_analysis.py     # 用户分析模块
│   └── portfolio_core.py    # 投资组合优化核心模块
├── Explainable agent/       # 可解释性与投后管理模块（解决痛点4-5）
│   ├── agent.py             # 投后陪伴顾问Agent核心实现
│   └── search.py            # 市场信息搜索模块
└── README.md                # 项目说明文档
```

## 核心功能模块

### 1. portfolio_recommender（组合推荐模块）

**主要功能**：
- 基于用户多维数据构建深度个性化投资组合
- 解决单品推荐到组合构建的鸿沟
- 支持动态画像与组合调整

**核心组件**：

#### PersonalizedPortfolioRecommender（主协调类）
- 负责串联整个推荐流程
- 整合数据预处理、用户分析、投资组合优化等模块
- 提供命令行接口，支持用户参数输入

```python
class PersonalizedPortfolioRecommender:
    def __init__(self, config=None):
        # 初始化各模块组件
        self.config = config or Config()
        self.data_preprocessor = DataPreprocessor()
        self.user_analyzer = UserProfileAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.risk_return_calculator = RiskReturnCalculator()
        self.result_processor = ResultProcessor()
    
    def recommend(self, user_data, product_data, target_return=None):
        # 完整推荐流程实现
```

#### DataPreprocessor（数据预处理模块）
- 处理用户和产品原始数据
- 支持多种数据格式的清洗和标准化
- 为后续分析提供结构化数据

#### ProductFilter（产品筛选器）
- 基于风险、投资期限和流动性要求筛选产品
- 帮助用户从众多产品中筛选出符合基本要求的候选池

#### UserProfileAnalyzer（用户画像分析器）
- 计算用户风险厌恶系数和目标收益率
- 考虑风险偏好、收入稳定性、年龄和投资期限等多维因素
- 构建动态更新的用户画像

```python
class UserProfileAnalyzer:
    def calculate_risk_aversion(self, user_data):
        # 计算风险厌恶系数
    
    def calculate_target_return(self, user_data):
        # 计算目标收益率
    
    def analyze_user_profile(self, user_data):
        # 综合分析用户画像
```

#### PortfolioOptimizer（投资组合优化器）
- 实现马科维茨投资组合优化模型
- 构建基于用户年龄、投资目标和流动性的个性化约束
- 提供多种优化目标选项（最大夏普比率、最小风险等）

```python
class PortfolioOptimizer:
    def build_constraints(self, user_data, product_data):
        # 构建个性化约束条件
    
    def optimize(self, risk_return_data, constraints):
        # 执行投资组合优化
```

#### RiskReturnCalculator（风险收益计算器）
- 计算产品间的协方差矩阵
- 评估组合的预期收益率、风险和夏普比率

#### ResultProcessor（结果处理器）
- 处理推荐结果，确保多样性和个性化程度
- 生成最终的投资组合建议和权重分配

### 2. Explainable agent（可解释性与投后管理模块）

**主要功能**：
- 为投资组合推荐提供透明、易懂的解释
- 提供持续的投后跟踪和市场分析
- 实现个性化的投资者教育和情绪管理

**核心组件**：

#### 投后陪伴顾问Agent
- 基于大语言模型和RAG技术实现
- 提供多维度的组合表现解释
- 生成市场情绪引导和投资教育内容

```python
def generate_explainability(portfolio, user_profile):
    # 生成组合表现、目标相关性、风险水平等多维度解释

def generate_market_sentiment_guidance(portfolio, market_data):
    # 生成市场情绪引导

def generate_investor_education(user_profile, market_condition):
    # 生成个性化投后教育内容

def generate_portfolio_tracking_summary(portfolio_performance):
    # 生成组合跟踪摘要，包含表现跟踪、关注资产和再平衡建议
```

#### 市场信息搜索功能
- 调用百度千帆AI搜索API获取最新市场热点和新闻
- 为投后陪伴提供实时的市场背景信息

## 数据结构

### 用户数据（profile.csv）
包含用户的多维特征信息：
- `user_id`: 用户唯一标识
- `age`: 年龄
- `occupations`: 职业
- `location`: 所在地
- `risk_profile`: 风险偏好（保守、稳健、激进等）
- `marital`: 婚姻状况
- `consumption_record`: 消费记录（JSON格式）
- `income_pattern`: 收入模式（三个月收入数组）
- `transaction_frequency`: 交易频率
- `holding_pattern`: 持仓模式（JSON数组）
- `survey_text`: 调查问卷文本
- `goal_type`: 投资目标类型
- `investment_horizon`: 投资期限（年）

### 产品数据（product.csv）
包含金融产品的详细信息：
- `product_id`: 产品唯一标识
- `产品名称`: 产品名称
- `产品类型`: 债券型、混合型、股票型、货币型等
- `预期收益率`: 年化预期收益率
- `波动率`: 历史波动率
- `风险等级`: R1-R5风险等级
- `夏普比率`: 风险调整后收益
- `最大回撤`: 历史最大回撤
- `最小投资期限(年)`: 最小持有期
- `流动性`: 高、中、低
- `资产类别`: 固定收益、权益、现金等价物、另类资产等

## 技术架构

### 1. 组合推荐引擎架构
- **数据层**：存储和管理用户数据、产品数据
- **分析层**：包含数据预处理、用户分析、风险收益计算等模块
- **优化层**：实现投资组合优化算法，构建个性化约束
- **输出层**：生成最终的组合建议和权重分配

### 2. 可解释性与投后管理架构
- **数据接入层**：获取实时市场数据和用户组合表现
- **分析处理层**：基于RAG技术的知识库构建和检索
- **生成层**：利用大语言模型生成解释、建议和教育内容
- **交互层**：提供用户友好的界面展示投后分析和建议

## 系统优势

### 1. 深度个性化推荐
- 超越简单的标签化推荐，结合用户多维数据
- 考虑人生阶段、收入稳定性等因素
- 与具体可量化的投资目标紧密结合

### 2. 组合构建的自动化与智能化
- 解决单品推荐到组合构建的鸿沟
- 自动完成复杂的资产配置和权重计算
- 提供科学的投资组合优化建议

### 3. 动态适应与更新
- 支持用户画像的动态更新
- 响应市场变化和用户需求变化
- 提供组合再平衡建议

### 4. 全面的透明度与可解释性
- 为每个推荐决策提供清晰的理由
- 解释投资逻辑、风险水平和预期收益
- 增强用户对推荐的信任度

### 5. 持续的投后陪伴与教育
- 提供组合表现的持续跟踪和解读
- 在不同市场环境下提供针对性的情绪管理
- 个性化的投资者教育内容

## 使用说明

### 运行投资组合推荐

```bash
cd portfolio_recommender
# 原始模型
python portfolio_recommender/main.py --users-csv "data/profile.csv" --products-csv "data/product.csv" --target-return-csv "data/target_return_from_llm.csv" --output-csv "data/result_1112.csv"

# 漂移模型：
python portfolio_recommender/main.py --users-csv "data/drift_profile.csv" --products-csv "data/product.csv" --target-return-csv "data/drift_target_return_from_llm.csv" --output-csv "data/dirft_result_1112.csv"

# 可加（推荐8个产品，最小权重0.03，指定编码）
  --max-products 8 
  --min-weight 0.03 
```

参数说明：
- `--user_id`: 指定用户ID
- `--data_path`: 数据文件夹路径
- `--output_path`: 结果输出路径

### 启动投后陪伴顾问

```bash
cd explainable_agent
python agent.py
```

## 未来展望

1. **多模态交互**：整合语音、图文等多种交互方式
2. **社交投资功能**：引入群组讨论和投资经验分享
3. **AI市场预测**：基于机器学习提升市场趋势预测准确性
4. **智能再平衡提醒**：根据市场变化和用户目标自动触发再平衡建议
5. **跨平台集成**：支持与主流金融App和交易平台的无缝对接

## 结语

本系统通过整合先进的金融工程理论和人工智能技术，致力于打造一个真正以用户为中心的智能投资顾问平台。我们不仅提供个性化的投资组合推荐，更注重全程陪伴用户的投资旅程，帮助用户在复杂多变的金融市场中做出更明智、更理性的投资决策。
