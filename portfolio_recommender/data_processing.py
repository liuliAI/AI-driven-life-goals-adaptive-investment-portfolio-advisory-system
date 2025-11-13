import pandas as pd
from typing import List, Optional
from config import DEFAULT_TOP_K_CANDIDATES


class DataPreprocessor:
    """数据预处理模块：统一用户/产品数据格式、字段映射"""
    
    @staticmethod
    def process_user_data(users_df: pd.DataFrame) -> pd.DataFrame:
        processed = users_df.copy()
        if 'user_id' in processed.columns:
            processed = processed.set_index('user_id')
        if 'investment_horizon' in processed.columns:
            processed['investment_horizon'] = pd.to_numeric(processed['investment_horizon'])
        if 'income_pattern' in processed.columns:
            processed['income_pattern'] = processed['income_pattern'].apply(
                lambda x: DataPreprocessor._parse_income_pattern(x)
            )
        return processed
    
    @staticmethod
    def process_product_data(products_df: pd.DataFrame) -> pd.DataFrame:
        processed = products_df.copy()
        field_mapping = {
            '最小投资期限(年)': 'min_period',
            '资产类别': 'asset_class',
            '预期收益率': 'expected_return',
            '风险等级': 'risk_level',
            '波动率': 'volatility',
            '夏普比率': 'sharpe_ratio',
            '流动性': 'liquidity'
        }
        processed = processed.rename(columns=field_mapping)
        numeric_fields = ['expected_return', 'volatility', 'sharpe_ratio', 'min_period']
        for field in numeric_fields:
            if field in processed.columns:
                processed[field] = pd.to_numeric(processed[field])
        return processed

    @staticmethod
    def _parse_income_pattern(income_str: Optional[str]) -> List[float]:
        if not income_str or pd.isna(income_str):
            print(f"警告：收入序列为空，使用默认值 [5000, 6000, 7000]")
            return [5000.0, 6000.0, 7000.0]
        try:
            if income_str.startswith('[') and income_str.endswith(']'):
                income_str = income_str[1:-1]
            return [float(num.strip()) for num in income_str.split(',')]
        except Exception as e:
            print(f"警告：收入序列解析失败 - {income_str}，使用默认值 [5000, 6000, 7000]")
            return [5000.0, 6000.0, 7000.0]


class ProductFilter:
    """产品筛选模块：基于用户特征筛选候选产品"""
    
    @staticmethod
    def filter_by_risk(risk_profile: str, products_df: pd.DataFrame) -> pd.DataFrame:
        risk_mapping = {'保守型': ['R1', 'R2'], '稳健型': ['R2', 'R3'], '进取型': ['R3', 'R4', 'R5']}
        target_risks = risk_mapping.get(risk_profile, ['R2', 'R3'])
        return products_df[products_df['risk_level'].isin(target_risks)]
    
    @staticmethod
    def filter_by_horizon(horizon: float, products_df: pd.DataFrame) -> pd.DataFrame:
        if horizon == -1:
            return products_df
        elif horizon <= 1:
            return products_df[products_df['min_period'] <= 1]
        elif horizon <= 3:
            return products_df[products_df['min_period'] <= 3]
        elif horizon <= 5:
            return products_df[products_df['min_period'] <= 5]
        else:
            return products_df
    
    @staticmethod
    def filter_by_liquidity(user_age: int, transaction_freq: float, products_df: pd.DataFrame) -> pd.DataFrame:
        if transaction_freq > 0.7 or user_age < 35:
            return products_df[products_df['liquidity'] == '高']
        elif transaction_freq > 0.4:
            return products_df[products_df['liquidity'].isin(['高', '中'])]
        else:
            return products_df
    
    @staticmethod
    def merge_filters(*filtered_dfs: pd.DataFrame) -> pd.DataFrame:
        if not filtered_dfs:
            return pd.DataFrame()
        common_index = set(filtered_dfs[0].index)
        for df in filtered_dfs[1:]:
            common_index.intersection_update(set(df.index))
        return filtered_dfs[0].loc[list(common_index)] if common_index else pd.DataFrame()
    
    @staticmethod
    def rank_by_sharpe(products_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        if len(products_df) <= top_k:
            return products_df
        return products_df.sort_values('sharpe_ratio', ascending=False).iloc[:top_k]
    
    def get_candidates(self, user_data: pd.Series, products_df: pd.DataFrame, top_k: int = DEFAULT_TOP_K_CANDIDATES) -> pd.DataFrame:
        risk_filtered = self.filter_by_risk(user_data['risk_profile'], products_df)
        horizon_filtered = self.filter_by_horizon(user_data['investment_horizon'], products_df)
        liquidity_filtered = self.filter_by_liquidity(user_data['age'], user_data['transaction_frequency'], products_df)
        candidates = self.merge_filters(risk_filtered, horizon_filtered, liquidity_filtered)
        return self.rank_by_sharpe(candidates, top_k)