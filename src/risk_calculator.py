import pandas as pd
import numpy as np
import yaml

class RiskCalculator:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['risk']

    def _resolve_severity_mapping(self, severity_scheme):
        """解析严重度量表：支持 1D (严重度->分值) 或 2D (严重度->后果类型->分值)。"""
        schemes = self.config.get('severity_schemes')
        if severity_scheme is not None and schemes and severity_scheme in schemes:
            return schemes[severity_scheme]
        # 回退到旧的单一 severity_mapping
        return self.config.get('severity_mapping', {})

    def _apply_severity(self, df, mapping):
        """根据严重度量表计算 severity_score (支持 1D / 2D 映射)。"""
        if not mapping:
            return df['Consequence severity'].map(lambda x: 0.0)
        first = next(iter(mapping.values()))
        if isinstance(first, dict):
            # 2D: 严重度 -> 后果类型 -> 分值
            def lookup(row):
                sev = row['Consequence severity']
                typ = row['Consequence type']
                level = mapping.get(sev, {})
                if isinstance(level, dict):
                    return level.get(typ, level.get('default', 0.0))
                return level
            return df.apply(lookup, axis=1)
        return df['Consequence severity'].map(lambda x: mapping.get(x, 0.0))

    def calculate_risk(self, df, severity_scheme=None):
        """
        根据 Risk = P * S 计算风险
        P (可能性): 基于聚类的频率权重
        S (严重度): 由 severity_scheme 指定的后果严重度量表给出
        """
        # 1. 计算严重度 S
        s_mapping = self._resolve_severity_mapping(severity_scheme)
        df['severity_score'] = self._apply_severity(df, s_mapping)
        
        # 2. 计算概率权重 P (基于频率)
        cluster_counts = df['cluster'].value_counts(normalize=True)
        df['p_weight'] = df['cluster'].map(cluster_counts)

        # 3. 计算风险 Risk = P * S
        df['risk'] = df['p_weight'] * df['severity_score']
        
        # 4. 执行 Min-Max 归一化到 [0, 1] 范围，避免 round(3) 吞掉过小的精度
        risk_min = df['risk'].min()
        risk_max = df['risk'].max()
        if risk_max > risk_min:
            df['risk'] = (df['risk'] - risk_min) / (risk_max - risk_min)
        else:
            df['risk'] = 0.0 # 或者保持原样，如果所有风险值都相同
            
        # 5. 指标保留 3 位小数 (用户偏好)
        df['risk'] = df['risk'].round(3)
        
        return df

    def get_risk_levels(self, df, bins=None, labels=None):
        """
        对风险值进行分级
        支持两种模式：
        1. 'rank': 基于百分比排名 (Quantile)，确保分箱数量均匀
        2. 'value': 基于真实风险分值 (Threshold)，反映绝对风险水平
        """
        method = self.config.get('binning_method', 'rank')
        
        if bins is None:
            bins = self.config.get('risk_bins', [0, 0.33, 0.66, 1.0])
            labels = self.config.get('risk_labels', ['Low', 'Medium', 'High'])
        elif labels is None:
            labels = ['Very low', 'Low', 'Medium', 'High', 'Very high'][:len(bins)-1]
        
        q_bins = [min(max(b, 0), 1.0) for b in bins]
        
        if method == 'rank':
            # 模式 1: 基于排名 (确保均匀)
            # 使用 method='first' 强制为重复值分配唯一排名
            df['risk_rank'] = df['risk'].rank(pct=True, method='average')
            df['Risk bin'] = pd.cut(
                df['risk_rank'], 
                bins=q_bins, 
                labels=labels, 
                include_lowest=True
            )
        else:
            # 模式 2: 基于真实值 (反映绝对水平)
            # 直接对归一化后的 risk [0, 1] 进行分箱
            df['Risk bin'] = pd.cut(
                df['risk'], 
                bins=q_bins, 
                labels=labels, 
                include_lowest=True
            )
            # 为了保持列一致性，提供一个空的 rank
            df['risk_rank'] = df['risk']
            
        return df

    def compare_binning_schemes(self, df, schemes):
        """
        对比多种风险等级划分方案，返回各等级的事件数量与占比。
        schemes: list of dict，每个含 {name, bins, labels}
        返回 (count_df, prop_df)：行=方案，列=风险等级
        """
        method = self.config.get('binning_method', 'rank')
        if method == 'rank':
            col = df['risk'].rank(pct=True, method='average')
        else:
            col = df['risk']

        counts = {}
        props = {}
        for s in schemes:
            q_bins = [min(max(b, 0), 1.0) for b in s['bins']]
            cats = pd.cut(col, bins=q_bins, labels=s['labels'], include_lowest=True)
            vc = cats.value_counts().reindex(s['labels']).fillna(0).astype(int)
            counts[s['name']] = vc
            props[s['name']] = vc / vc.sum() if vc.sum() > 0 else vc * 0.0

        count_df = pd.DataFrame(counts).T
        prop_df = pd.DataFrame(props).T
        return count_df, prop_df
