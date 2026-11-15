import pandas as pd
import numpy as np
import yaml
import os

class DataProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
    def load_data(self, file_path=None):
        """加载原始 CSV 数据"""
        if file_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(base_dir, self.config['data']['input_path'])
        df = pd.read_csv(file_path)
        return df

    def clean_data(self, df):
        """数据清洗与地理归并"""
        # 1. 只保留有效年份并转换为整数
        df = df[df['Year'].notna()].copy()
        df['Year'] = df['Year'].astype(int)
        
        # 2. 地理归并 (规范化名称并应用映射)
        geo_merge = self.config['data']['geographic_merge']
        for col in ['Attack countries', 'Victim countries']:
            if col in df.columns:
                # 先标准化：去空格 -> Title Case
                df[col] = df[col].map(lambda x: str(x).strip().title() if pd.notnull(x) else x)
                # 再应用映射 (如 "Usa" -> "United States")
                df[col] = df[col].map(lambda x: geo_merge.get(x, x))
        
        return df

    def get_categorical_data(self, df):
        """提取用于聚类的分类变量"""
        cat_cols = self.config['data']['categorical_columns']
        df_filtered = df[df['Year'] >= 2010].copy()
        return df_filtered[cat_cols]
