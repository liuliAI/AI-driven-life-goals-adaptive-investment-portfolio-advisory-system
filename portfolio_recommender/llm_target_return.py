import pandas as pd
from openai import OpenAI
from typing import Dict, List, Tuple, Optional, Union, Any

# 初始化OpenAI客户端
client = OpenAI(
    api_key='sk-5d333xxxxxxexxxxxxxxxx',  #已脱敏，使用新的API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 用户数据文件路径
USERS_FILE_PATH = "C:\\Users\\szf\\Desktop\\金融科技大赛\\profile.csv"

DRIFT_FILE_PATH = "C:\\Users\\szf\\Desktop\\金融科技大赛\\drift_profile.csv"

target_return_path = 'C:\\Users\\szf\\Desktop\\金融科技大赛\\target_return_from_llm.csv'

drift_target_return_path = 'C:\\Users\\szf\\Desktop\\金融科技大赛\\drift_target_return_from_llm.csv'

IS_DRIFT = True

def generate_user_prompt(user_dict: Dict[str, Any]) -> str:
    
    prompt = f"""
请基于以下用户的财务信息及个人画像数据，测算其合理的预期收益率：
- 投资目标：{user_dict['survey_text']}
- 年龄：{user_dict['age']}
- 职业：{user_dict['occupations']}
- 所在城市：{user_dict['location']}
- 风险偏好：{user_dict['risk_profile']}
- 婚姻状况：{user_dict['marital']}
- 消费记录：{user_dict['consumption_record']}
- 近期收入模式：{user_dict['income_pattern']}
- 交易频率：{user_dict['transaction_frequency']}
- 持仓模式：{user_dict['holding_pattern']}
- 账户余额：{user_dict['Account Balance']}

测算要求：
1. 以投资目标和用户画像信息为核心依据，在风险可控且与用户风险偏好匹配的前提下，尽可能贴合用户投资目标；
2. 若投资目标与风险偏好存在冲突，需以风险偏好为基础，对预期收益率进行合理调整（即允许投资目标适度妥协）；
3. 输出格式：仅返回浮点数（例如：若预期收益率为5%，则返回5.0），无需任何额外内容。
"""
    return prompt.strip()


def generate_drift_prompt(user_dict: Dict[str, Any]) -> str:
    """根据用户数据生成测算预期收益率的prompt"""
    prompt = f"""
请根据用户原始目标和用户现状，测算其合理的预期收益率：
用户的原始投资目标为：
- 投资目标：{user_dict['survey_text']}
经过了{user_dict['drift years']}年后，用户的现状为：
- 年龄：{user_dict['age']}
- 职业：{user_dict['occupations']}
- 所在城市：{user_dict['location']}
- 风险偏好：{user_dict['risk_profile']}
- 婚姻状况：{user_dict['marital']}
- 消费记录：{user_dict['consumption_record']}
- 近期收入模式：{user_dict['income_pattern']}
- 交易频率：{user_dict['transaction_frequency']}
- 持仓模式：{user_dict['holding_pattern']}
- 账户余额：{user_dict['Account Balance']}

测算要求：
1. 以投资目标和用户当前画像信息为核心依据，在风险可控且与用户风险偏好匹配的前提下，尽可能贴合用户投资目标；
2. 输出格式：仅返回浮点数（例如：若预期收益率为5%，则返回5.0），无需任何额外内容。
"""
    return prompt.strip()


if __name__ == "__main__":

    if not IS_DRIFT:
        # 读取用户数据
        users_df = pd.read_csv(USERS_FILE_PATH)
        results = []
        
        for _, row in users_df.iterrows():
            # 处理空值并转换为字典
            user_dict = row.fillna("无").to_dict()
            
            # 生成提示词并调用模型
            prompt = generate_user_prompt(user_dict)
            completion = client.chat.completions.create(
                model='qwen3-max',
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},
                stream=False
            )
            
            # 提取模型回复
            target_return = completion.choices[0].message.content
            print(f"用户 {user_dict['user_id']} 模型回复: {target_return}")
            
            # 保存结果
            results.append({
                "user_id": user_dict["user_id"],
                "target_return": target_return
            })
            
        # 导出结果到CSV
        pd.DataFrame(results).to_csv("target_return_from_llm.csv", index=False)
        print("\n所有用户测算完成，结果已保存至 target_return_from_llm.csv")
    else:
        # 漂移模型
        users_df = pd.read_csv(USERS_FILE_PATH, usecols = ['user_id', 'survey_text', 'investment_horizon'], encoding='gbk')
        drift_df = pd.read_csv(DRIFT_FILE_PATH)
        target_return_df = pd.read_csv(target_return_path)

        users_df = pd.merge(users_df, target_return_df, on='user_id', how='inner')

        all_df = pd.merge(users_df, drift_df, on='user_id', how='inner')
        results = []
        
        for _, row in all_df.iterrows():
            # 处理空值并转换为字典
            user_dict = row.fillna("无").to_dict()
            
            if user_dict['investment_horizon'] == -1 or user_dict['investment_horizon'] <= user_dict['drift years']:
                results.append({
                    "user_id": user_dict["user_id"],
                    "target_return": user_dict['target_return']
                })
            else:
                prompt = generate_drift_prompt(user_dict)
                completion = client.chat.completions.create(
                    model='qwen3-max',
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"enable_thinking": False},
                    stream=False
                )
                
                # 提取模型回复
                target_return = completion.choices[0].message.content
                print(f"用户 {user_dict['user_id']} 模型回复: {target_return}")
                
                # 保存结果
                results.append({
                    "user_id": user_dict["user_id"],
                    "target_return": target_return
                })
            # break
            
        # 导出结果到CSV
        pd.DataFrame(results).to_csv("drift_target_return_path.csv", index=False)
        print("\n所有用户测算完成，结果已保存至 drift_target_return_path.csv")