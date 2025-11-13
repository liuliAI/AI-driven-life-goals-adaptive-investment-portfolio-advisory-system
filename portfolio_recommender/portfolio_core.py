import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
from typing import List, Dict, Optional
from config import DEFAULT_MIN_WEIGHT


class PortfolioOptimizer:
    """组合优化模块：约束构建 + 最优权重求解"""
    
    @staticmethod
    def build_age_constraints(age: int, products_df: pd.DataFrame) -> List[Dict]:
        """基于年龄的约束"""
        constraints = []
        if age < 30:  # 年轻人要求权益类占比
            equity_mask = products_df['asset_class'] == 'equities'
            if equity_mask.any():
                equity_indices = np.where(equity_mask)[0].tolist()
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: np.sum([x[i] for i in equity_indices]) - 0.3  # 至少30%权益
                })
        elif age > 60:  # 老年人限制高风险
            high_risk_mask = products_df['risk_level'].isin(['R4', 'R5'])
            if high_risk_mask.any():
                high_risk_indices = np.where(high_risk_mask)[0].tolist()
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: 0.2 - np.sum([x[i] for i in high_risk_indices])  # 高风险≤20%
                })
        return constraints
    
    @staticmethod
    def build_goal_constraints(goal_type: str, products_df: pd.DataFrame) -> List[Dict]:
        """基于投资目标的约束"""
        constraints = []
        if goal_type == '房产购置':  # 购房限制高风险
            high_risk_mask = products_df['risk_level'].isin(['R4', 'R5'])
            if high_risk_mask.any():
                high_risk_indices = np.where(high_risk_mask)[0].tolist()
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: 0.3 - np.sum([x[i] for i in high_risk_indices])  # 高风险≤30%
                })
        elif goal_type == '退休':  # 退休要求长期产品
            long_term_mask = products_df['min_period'] > 5
            if long_term_mask.any():
                long_term_indices = np.where(long_term_mask)[0].tolist()
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: np.sum([x[i] for i in long_term_indices]) - 0.4  # 长期≥40%
                })
        return constraints
    
    @staticmethod
    def build_liquidity_constraints(transaction_freq: float, products_df: pd.DataFrame) -> List[Dict]:
        """基于流动性的约束"""
        constraints = []
        if transaction_freq > 0.7:  # 高交易频率要求高流动性
            high_liquid_mask = products_df['liquidity'] == '高'
            if high_liquid_mask.any():
                high_liquid_indices = np.where(high_liquid_mask)[0].tolist()
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: np.sum([x[i] for i in high_liquid_indices]) - 0.5  # 高流动性≥50%
                })
        return constraints
    
    def build_constraints(self, user_data: pd.Series, products_df: pd.DataFrame) -> List[Dict]:
        """整合所有个性化约束"""
        constraints = []
        constraints.extend(self.build_age_constraints(user_data['age'], products_df))
        constraints.extend(self.build_goal_constraints(user_data['goal_type'], products_df))
        constraints.extend(self.build_liquidity_constraints(user_data['transaction_frequency'], products_df))
        return constraints
    
    @staticmethod
    def solve_optimization(expected_returns: np.ndarray, 
                          cov_matrix: np.ndarray,
                          risk_aversion: float, 
                          target_return: float, 
                          constraints: List[Dict]) -> np.ndarray:
        """求解马科维茨优化问题（最大化效用=收益-风险厌恶×风险）"""
        n_assets = len(expected_returns)
        if n_assets == 0:
            return np.array([])
        
        # 目标函数：最小化负效用
        def objective(weights):
            port_return = np.dot(weights, expected_returns)
            port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            return -(port_return - risk_aversion * port_var)
        
        # 初始权重与约束
        x0 = np.ones(n_assets) / n_assets
        constraints = constraints.copy()
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 权重和为1
        bounds = [(0, 1) for _ in range(n_assets)]  # 不允许卖空
        
        # 目标收益约束
        constraints.append({'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return})
        
        # 求解
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        if result.success:
            return result.x
        else:
            print(f"优化求解失败，返回等权重（错误：{result.message}）")
            return np.ones(n_assets) / n_assets  # 失败时返回等权重


class RiskReturnCalculator:
    """风险收益计算模块：协方差矩阵 + 组合收益/风险"""
    
    @staticmethod
    def calculate_covariance(products_df: pd.DataFrame) -> np.ndarray:
        """计算产品协方差矩阵（基于波动率和假设相关性）"""
        n = len(products_df)
        if n <= 1:
            return np.array([]) if n == 0 else np.array([[products_df['volatility'].iloc[0]**2]])
        
        volatilities = np.asarray(products_df['volatility'].values)  # 关键修改
        base_corr = 0.3  # 假设平均相关性
        # 构建相关性矩阵
        corr_matrix = np.eye(n) * (1 - base_corr) + base_corr
        # 协方差矩阵 = 波动率外积 × 相关性矩阵
        return np.outer(volatilities, volatilities) * corr_matrix
    
    @staticmethod
    def portfolio_return(recommendation: List[Dict]) -> float:
        """计算组合预期收益率"""
        return sum(p['expected_return'] * p['recommended_weight'] for p in recommendation)
    
    @staticmethod
    def portfolio_risk(recommendation: List[Dict]) -> float:
        """计算组合风险（波动率）"""
        if len(recommendation) <= 1:
            return recommendation[0]['volatility'] if recommendation else 0.0
        
        weights = np.array([p['recommended_weight'] for p in recommendation])
        vols = np.array([p['volatility'] for p in recommendation])
        avg_corr = 0.3  # 简化假设
        
        # 计算组合方差
        var = np.sum(weights**2 * vols**2)
        # 加入协方差项
        for i in range(len(weights)):
            for j in range(i+1, len(weights)):
                var += 2 * avg_corr * weights[i] * weights[j] * vols[i] * vols[j]
        return np.sqrt(var)


class ResultProcessor:
    """结果处理模块：后处理 + 推荐质量分析"""
    
    @staticmethod
    def post_process(products_df: pd.DataFrame, 
                    weights: np.ndarray, 
                    max_products: int, 
                    min_weight: float = DEFAULT_MIN_WEIGHT) -> List[Dict]:
        """后处理：筛选高权重产品并归一化权重"""
        if len(products_df) == 0 or len(weights) == 0:
            return []
        
        # 按权重排序
        sorted_indices = np.argsort(weights)[::-1]
        selected_indices = sorted_indices[:max_products]
        
        # 过滤低权重产品并归一化
        selected_weights = weights[selected_indices]
        valid_mask = selected_weights > min_weight  # 权重大于 min_weight 才保留
        selected_indices = selected_indices[valid_mask]
        selected_weights = selected_weights[valid_mask]
        
        if len(selected_weights) == 0:
            return []
        selected_weights /= np.sum(selected_weights)  # 重新归一化
        
        # 构建推荐结果
        recommendation = []
        for idx, w in zip(selected_indices, selected_weights):
            product = products_df.iloc[idx].to_dict()
            product['recommended_weight'] = round(w, 4)
            recommendation.append(product)
        return recommendation
    
    @staticmethod
    def calculate_diversity(recommendations: Dict) -> float:
        """计算推荐多样性得分（香农熵）"""
        product_counts = {}
        total = 0
        for rec in recommendations.values():
            for p in rec['recommended_products']:
                pid = p['product_id']
                product_counts[pid] = product_counts.get(pid, 0) + 1
                total += 1
        
        if total == 0:
            return 0.0
        probs = [c / total for c in product_counts.values()]
        max_ent = np.log(len(product_counts)) if product_counts else 0
        
        return float(entropy(probs) / max_ent) if max_ent > 0 else 0.0
    
    @staticmethod
    def calculate_personalization(recommendations: Dict) -> float:
        """计算个性化程度（1-用户间平均相似度）"""
        if len(recommendations) <= 1:
            return 1.0
        
        similarities = []
        user_ids = list(recommendations.keys())
        for i in range(len(user_ids)):
            for j in range(i+1, len(user_ids)):
                p1 = {p['product_id'] for p in recommendations[user_ids[i]]['recommended_products']}
                p2 = {p['product_id'] for p in recommendations[user_ids[j]]['recommended_products']}
                if not p1 or not p2:
                    continue
                intersection = len(p1 & p2)
                union = len(p1 | p2)
                similarities.append(intersection / union if union > 0 else 0)
        
        return 1 - float(np.mean(similarities)) if similarities else 1.0
    
    def analyze_recommendations(self, recommendations: Dict) -> Dict:
        """整合推荐质量分析结果"""
        if not recommendations:
            return {
                'avg_portfolio_size': 0, 'avg_expected_return': 0,
                'avg_expected_risk': 0, 'diversity_score': 0, 'personalization_score': 0
            }
        
        n_users = len(recommendations)
        avg_size = sum(len(rec['recommended_products']) for rec in recommendations.values()) / n_users
        avg_return = sum(rec['expected_return'] for rec in recommendations.values()) / n_users
        avg_risk = sum(rec['expected_risk'] for rec in recommendations.values()) / n_users
        
        return {
            'avg_portfolio_size': round(avg_size, 2),
            'avg_expected_return': round(avg_return, 4),
            'avg_expected_risk': round(avg_risk, 4),
            'diversity_score': round(self.calculate_diversity(recommendations), 4),
            'personalization_score': round(self.calculate_personalization(recommendations), 4)
        }