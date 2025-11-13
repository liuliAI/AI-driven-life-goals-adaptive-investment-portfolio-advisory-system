import os
import argparse
import traceback
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional  
import chardet

# 导入所有核心模块
from config import (
    DEFAULT_MAX_PRODUCTS, DEFAULT_MIN_WEIGHT
)
from data_processing import DataPreprocessor, ProductFilter
from user_analysis import UserProfileAnalyzer
from portfolio_core import PortfolioOptimizer, RiskReturnCalculator, ResultProcessor

def detect_file_encoding(file_path: str) -> str:
    """自动检测文件编码（优先utf-8-sig，其次gbk，最后默认utf-8）"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(1024 * 10)  # 读取前10KB
    
    result = chardet.detect(raw_data)
    if not result:
        return 'utf-8-sig'
    
    # 关键修复：过滤非字符串编码，确保类型符合 Optional[str]
    encoding = result.get('encoding')
    if not isinstance(encoding, str):  # 排除 float/None 等非字符串类型
        encoding = None
    
    confidence = result.get('confidence', 0.0)  # 置信度默认0.0（float类型安全）
    
    if confidence > 0.8 and encoding:
        encoding_lower = encoding.lower()
        if encoding_lower in ['utf-8', 'utf-8-sig']:
            return 'utf-8-sig'
        elif encoding_lower in ['gbk', 'gb2312', 'gb18030']:
            return 'gbk'
    
    return 'utf-8-sig'


class PersonalizedPortfolioRecommender:
    """主协调类：串联所有模块，执行完整推荐流程"""
    
    def __init__(self, users_df: pd.DataFrame, products_df: pd.DataFrame):
        # 初始化各核心模块
        self.data_preprocessor = DataPreprocessor()
        self.product_filter = ProductFilter()
        self.profile_analyzer = UserProfileAnalyzer()
        self.optimizer = PortfolioOptimizer()
        self.risk_calculator = RiskReturnCalculator()
        self.result_processor = ResultProcessor()
        
        # 预处理数据（统一格式、字段映射）
        self.users_data = self.data_preprocessor.process_user_data(users_df)
        self.products_data = self.data_preprocessor.process_product_data(products_df)
        
        print(f"✅ 推荐器初始化完成: {len(self.users_data)} 个用户, {len(self.products_data)} 个产品")
    
    def recommend_for_user(self, 
                          user_id: str, 
                          max_products: int = DEFAULT_MAX_PRODUCTS, 
                          min_weight: float = DEFAULT_MIN_WEIGHT) -> Dict:
        """为单个用户生成个性化投资组合推荐"""
        # 校验用户ID是否存在
        if user_id not in self.users_data.index:
            raise ValueError(f"用户 {user_id} 不存在于数据中")

        user_data = self.users_data.loc[user_id]
        if isinstance(user_data, pd.DataFrame):
            user_data = user_data.iloc[0]  # 取第一条记录
        user_data = pd.Series(user_data)

        # 1. 筛选候选产品（风险、期限、流动性三重过滤 + 夏普比率排序）
        candidates = self.product_filter.get_candidates(user_data, self.products_data)
        if len(candidates) == 0:
            print(f"⚠️ 用户 {user_id} 无符合条件的产品，返回空推荐")
            return {
                'recommended_products': [],
                'expected_return': 0.0,
                'expected_risk': 0.0,
                'personalized_aversion': 0.0
            }
        
        # 2. 计算用户个性化参数（风险厌恶系数、目标收益率）
        risk_aversion = self.profile_analyzer.calculate_risk_aversion(user_data)
        target_return = self.profile_analyzer.calculate_target_return(user_data)
        
        # 3. 构建个性化约束 + 求解最优权重
        constraints = self.optimizer.build_constraints(user_data, candidates)
        expected_returns = np.asarray(candidates['expected_return'].values)
        cov_matrix = self.risk_calculator.calculate_covariance(candidates)
        optimal_weights = self.optimizer.solve_optimization(
            expected_returns, cov_matrix, risk_aversion, target_return, constraints
        )
        
        # 4. 后处理推荐结果（过滤低权重、归一化、限制产品数量）
        recommendation = self.result_processor.post_process(
            candidates, optimal_weights, max_products, min_weight
        )
        
        # 5. 计算组合最终收益与风险
        port_return = self.risk_calculator.portfolio_return(recommendation)
        port_risk = self.risk_calculator.portfolio_risk(recommendation)
        
        return {
            'recommended_products': recommendation,
            'expected_return': port_return,
            'expected_risk': port_risk,
            'personalized_aversion': risk_aversion
        }
    
    def run_pipeline(self, max_products_per_user: int = DEFAULT_MAX_PRODUCTS) -> Tuple[Dict, Dict]:
        """运行完整推荐流程：为所有用户生成推荐 + 分析推荐质量"""
        all_recommendations = {}
        total_users = len(self.users_data)
        
        for idx, user_id in enumerate(self.users_data.index, 1):
            # 每处理10个用户输出进度
            if idx % 10 == 0 or idx == total_users:
                print(f"进度：{idx}/{total_users} 个用户")
            
            try:
                all_recommendations[user_id] = self.recommend_for_user(
                    user_id, max_products_per_user
                )
            except Exception as e:
                print(f"处理用户 {user_id} 失败：{str(e)}，跳过该用户")
        
        # 分析推荐质量（多样性、个性化、收益风险等）
        analysis_results = self.result_processor.analyze_recommendations(all_recommendations)
        return all_recommendations, analysis_results


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="个性化投资组合推荐系统",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 必选参数
    parser.add_argument(
        "--users-csv", 
        required=True,
        help="用户数据CSV路径（必须包含：user_id、risk_profile、age、investment_horizon等字段）"
    )
    parser.add_argument(
        "--products-csv", 
        required=True,
        help="产品数据CSV路径（必须包含：product_id、资产类别、预期收益率、风险等级等字段）"
    )
    parser.add_argument(
        "--target-return-csv", 
        required=True,
        help="用户目标收益率CSV路径（必须包含：user_id、target_return字段）"
    )
    parser.add_argument(
        "--output-csv", 
        required=True,
        help="推荐结果输出CSV路径（包含：user_id、product_id、推荐权重等信息）"
    )
    
    # 可选参数（推荐规则配置）
    parser.add_argument(
        "--max-products", 
        type=int, 
        default=DEFAULT_MAX_PRODUCTS,
        help=f"每个用户最多推荐产品数量（默认：{DEFAULT_MAX_PRODUCTS}）"
    )
    parser.add_argument(
        "--min-weight", 
        type=float, 
        default=DEFAULT_MIN_WEIGHT,
        help=f"产品最小推荐权重（低于此值不推荐，默认：{DEFAULT_MIN_WEIGHT}）"
    )
    

    return parser.parse_args()


def main():
    """主函数：参数解析 → 数据读取 → 推荐执行 → 结果输出"""
    args = parse_args()
    
    try:
        
        print(f"\n正在读取数据...")
        user_encoding = detect_file_encoding(args.users_csv)
        product_encoding = detect_file_encoding(args.products_csv)
        target_return_encoding = detect_file_encoding(args.target_return_csv)

        users_df = pd.read_csv(args.users_csv, encoding=user_encoding)
        target_return_df = pd.read_csv(args.target_return_csv, encoding=target_return_encoding)
        
        # 合并用户数据和目标收益率（按user_id内连接）
        users_df = pd.merge(users_df, target_return_df, on='user_id', how='inner')
        print(f"用户数据合并完成：{len(users_df)} 条有效用户记录")
        
        # 读取产品数据
        products_df = pd.read_csv(args.products_csv, encoding=product_encoding)
        print(f"产品数据读取完成：{len(products_df)} 条产品记录")
        
        # 2. 初始化推荐器并运行完整流程
        recommender = PersonalizedPortfolioRecommender(users_df, products_df)
        print(f"\n开始生成个性化推荐...")
        all_recommendations, analysis_results = recommender.run_pipeline(
            max_products_per_user=args.max_products
        )
        
        # 3. 输出推荐质量分析报告
        print(f"\n推荐质量分析报告")
        print("-" * 50)
        print(f"平均组合产品数量：{analysis_results['avg_portfolio_size']}")
        print(f"平均预期收益率：{analysis_results['avg_expected_return']:.2%}")
        print(f"平均预期风险（波动率）：{analysis_results['avg_expected_risk']:.2%}")
        print(f"推荐多样性得分：{analysis_results['diversity_score']}（越高越多样）")
        print(f"个性化程度得分：{analysis_results['personalization_score']}（越高越个性化）")
        print("-" * 50)
        
        # 4. 保存推荐结果到CSV
        result_rows = []
        for user_id, rec in all_recommendations.items():
            for product in rec['recommended_products']:
                result_rows.append({
                    'user_id': user_id,
                    'product_id': product.get('product_id', ''),
                    'asset_class': product.get('asset_class', ''),
                    'risk_level': product.get('risk_level', ''),
                    'expected_return': product.get('expected_return', 0.0),
                    'volatility': product.get('volatility', 0.0),
                    'recommended_weight': product.get('recommended_weight', 0.0),
                    'personalized_risk_aversion': rec['personalized_aversion']
                })
        
        result_df = pd.DataFrame(result_rows)
        result_df.to_csv(args.output_csv, index=False, encoding=args.encoding_output)
        print(f"\n推荐结果已保存到：{os.path.abspath(args.output_csv)}")
        
    except FileNotFoundError as e:
        print(f"\n错误：找不到文件 - {e.filename}")
        print("请检查文件路径是否正确，或文件是否存在")
    except KeyError as e:
        print(f"\n错误：数据缺少必要字段 - {e}")
        print("请检查CSV文件是否包含所有必需字段（参考--help说明）")
    except Exception as e:
        print(f"\n意外错误：{type(e).__name__} - {str(e)}")
        print("\n详细错误追踪：")
        traceback.print_exc()


if __name__ == "__main__":
    main()