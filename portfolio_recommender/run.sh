# 原始模型

python portfolio_recommender/main.py --users-csv "data/profile.csv" --products-csv "data/product.csv" --target-return-csv "data/target_return_from_llm.csv" --output-csv "data/result_1112.csv"


# 漂移模型：
python portfolio_recommender/main.py --users-csv "data/drift_profile.csv" --products-csv "data/product.csv" --target-return-csv "data/drift_target_return_from_llm.csv" --output-csv "data/dirft_result_1112.csv"

# 可加（推荐8个产品，最小权重0.03，指定编码）
  --max-products 8 
  --min-weight 0.03 
