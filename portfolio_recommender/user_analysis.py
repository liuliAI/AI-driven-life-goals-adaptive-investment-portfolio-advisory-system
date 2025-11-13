import pandas as pd
import numpy as np
from typing import List


class UserProfileAnalyzer:
    """用户画像分析模块：计算风险厌恶系数、目标收益率"""
    
    @staticmethod
    def map_risk_to_aversion(risk_profile: str) -> float:
        risk_mapping = {'保守型': 4.0, '稳健型': 2.0, '进取型': 1.0}
        return risk_mapping.get(risk_profile, 2.0)
    
    @staticmethod
    def calculate_income_stability(income_pattern: List[float]) -> float:
        if len(income_pattern) < 3:
            return 0.5
        valid_incomes = [inc for inc in income_pattern if 0 < inc < 1000000]
        if len(valid_incomes) < 2:
            return 0.3
        incomes = np.array(valid_incomes)
        median = np.median(incomes)
        if median == 0:
            return 0.0
        mad = np.mean(np.abs(incomes - median))
        cv = mad / median
        return np.clip(1 - (cv - 0.1) / 0.4, 0.0, 1.0).item() if 0.1 < cv < 0.5 else 1.0 if cv <= 0.1 else 0.0
    
    def _age_factor(self, age: int, conservative: bool) -> float:
        if conservative:
            return 0.9 if age <=35 else 1.0 if age <=50 else 1.1 if age <=65 else 1.2
        else:
            return 0.5 if age <=25 else 0.7 if age <=35 else 1.0 if age <=45 else 1.3 if age <=55 else 1.6
    
    def _horizon_factor(self, horizon: float, conservative: bool) -> float:
        if horizon == -1:
            return 1.0
        if conservative:
            return 1.2 if horizon <=1 else 1.1 if horizon <=3 else 1.0 if horizon <=5 else 0.9 if horizon <=10 else 0.8
        else:
            return 0.7 if horizon <=1 else 0.8 if horizon <=3 else 1.0 if horizon <=5 else 1.2 if horizon <=10 else 1.4
    
    def _stability_factor(self, income_pattern: List[float]) -> float:
        return 0.8 + self.calculate_income_stability(income_pattern) * 0.4
    
    def calculate_risk_aversion(self, user_data: pd.Series) -> float:
        base_aversion = self.map_risk_to_aversion(user_data['risk_profile'])
        age = user_data['age']
        horizon = user_data['investment_horizon']
        income_pattern = user_data['income_pattern']
        
        if base_aversion >= 3.5:
            factor = self._age_factor(age, True) * self._horizon_factor(horizon, True) * self._stability_factor(income_pattern)
            return np.clip(base_aversion * factor, 3.0, 5.0)
        elif base_aversion >= 1.5:
            factor = self._age_factor(age, False) / self._horizon_factor(horizon, False) * self._stability_factor(income_pattern)
            return np.clip(base_aversion * factor, 1.0, 4.0)
        else:
            factor = self._age_factor(age, False) / (self._horizon_factor(horizon, False) * 1.2) * self._stability_factor(income_pattern)
            return np.clip(base_aversion * factor, 0.5, 2.5)
    
    def calculate_target_return(self, user_data: pd.Series) -> float:
        return user_data['target_return'] / 100 + 0.03