import pandas as pd
import numpy as np
import yaml

class RiskCalculator:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['risk']

    def _resolve_severity_mapping(self, severity_scheme):
        """ 1D (->)  2D (->->)"""
        schemes = self.config.get('severity_schemes')
        if severity_scheme is not None and schemes and severity_scheme in schemes:
            return schemes[severity_scheme]
        
        return self.config.get('severity_mapping', {})

    def _apply_severity(self, df, mapping):
        """ severity_score ( 1D / 2D )"""
        if not mapping:
            return df['Consequence severity'].map(lambda x: 0.0)
        first = next(iter(mapping.values()))
        if isinstance(first, dict):
            
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
         Risk = P * S 
        P (): 
        S ():  severity_scheme 
        """
        
        s_mapping = self._resolve_severity_mapping(severity_scheme)
        df['severity_score'] = self._apply_severity(df, s_mapping)
        
        
        cluster_counts = df['cluster'].value_counts(normalize=True)
        df['p_weight'] = df['cluster'].map(cluster_counts)

        
        df['risk'] = df['p_weight'] * df['severity_score']
        
        
        risk_min = df['risk'].min()
        risk_max = df['risk'].max()
        if risk_max > risk_min:
            df['risk'] = (df['risk'] - risk_min) / (risk_max - risk_min)
        else:
            df['risk'] = 0.0 
            
        
        df['risk'] = df['risk'].round(3)
        
        return df

    def get_risk_levels(self, df, bins=None, labels=None):
        """
        
        
        1. 'rank':  (Quantile)
        2. 'value':  (Threshold)
        """
        method = self.config.get('binning_method', 'rank')
        
        if bins is None:
            bins = self.config.get('risk_bins', [0, 0.33, 0.66, 1.0])
            labels = self.config.get('risk_labels', ['Low', 'Medium', 'High'])
        elif labels is None:
            labels = ['Very low', 'Low', 'Medium', 'High', 'Very high'][:len(bins)-1]
        
        q_bins = [min(max(b, 0), 1.0) for b in bins]
        
        if method == 'rank':
            
            
            df['risk_rank'] = df['risk'].rank(pct=True, method='average')
            df['Risk bin'] = pd.cut(
                df['risk_rank'], 
                bins=q_bins, 
                labels=labels, 
                include_lowest=True
            )
        else:
            
            
            df['Risk bin'] = pd.cut(
                df['risk'], 
                bins=q_bins, 
                labels=labels, 
                include_lowest=True
            )
            
            df['risk_rank'] = df['risk']
            
        return df

    def compare_binning_schemes(self, df, schemes):
        """
        
        schemes: list of dict {name, bins, labels}
         (count_df, prop_df)==
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
