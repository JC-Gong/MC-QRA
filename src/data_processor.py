import pandas as pd
import numpy as np
import yaml
import os

class DataProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
    def load_data(self, file_path=None):
        """ CSV """
        if file_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(base_dir, self.config['data']['input_path'])
        df = pd.read_csv(file_path)
        return df

    def clean_data(self, df):
        """"""
        
        df = df[df['Year'].notna()].copy()
        df['Year'] = df['Year'].astype(int)
        
        
        geo_merge = self.config['data']['geographic_merge']
        for col in ['Attack countries', 'Victim countries']:
            if col in df.columns:
                
                df[col] = df[col].map(lambda x: str(x).strip().title() if pd.notnull(x) else x)
                
                df[col] = df[col].map(lambda x: geo_merge.get(x, x))
        
        return df

    def get_categorical_data(self, df):
        """"""
        cat_cols = self.config['data']['categorical_columns']
        df_filtered = df[df['Year'] >= 2010].copy()
        return df_filtered[cat_cols]
