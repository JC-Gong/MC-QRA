import os
import sys
import pandas as pd
import numpy as np
import yaml


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_severity_mapping(risk_cfg):
    """优先使用 severity_schemes，否则回退到单一 severity_mapping。"""
    schemes = risk_cfg.get('severity_schemes')
    if schemes:
        # 无显式 scheme 时取第一个
        first = next(iter(schemes.values()))
        return first
    return risk_cfg.get('severity_mapping', {})


def build_cluster_risk_table(df, severity_map, cluster_col='cluster',
                             severity_col='Consequence severity'):
    """对聚类结果逐簇统计风险指标。

    指标定义 (对齐论文口径):
      Incidents      : 该簇事件数
      Frequency      : Incidents / 总事件数
      Mean severity  : 该簇 severity_score 均值
      Relative risk  : Frequency * Mean severity
      Risk share     : Relative risk / Sum(Relative risk)
      Rank           : 按 Relative risk 降序排名 (1 = 最高)
    """
    df = df.copy()
    df['severity_score'] = df[severity_col].map(
        lambda x: severity_map.get(x, np.nan)
    )

    total = len(df)
    g = df.groupby(cluster_col, observed=False)

    rows = []
    for cluster_id, sub in g:
        n = len(sub)
        mean_sev = sub['severity_score'].mean()
        freq = n / total
        rel_risk = freq * mean_sev
        rows.append({
            'cluster': int(cluster_id),
            'Incidents': int(n),
            'Frequency': freq,
            'Mean severity': mean_sev,
            'Relative risk': rel_risk,
            'Risk share': rel_risk,  # 占位，下面归一
        })

    table = pd.DataFrame(rows)
    table['Risk share'] = table['Relative risk'] / table['Relative risk'].sum()

    # 排名 (Relative risk 降序, 1 为最高)
    table = table.sort_values('cluster').reset_index(drop=True)
    table['Rank'] = table['Relative risk'].rank(ascending=False, method='min').astype(int)

    # 重命名场景列
    table.insert(0, 'Scenario', ['C' + str(i + 1) for i in range(len(table))])

    # 总计行
    total_row = {
        'Scenario': 'Total',
        'cluster': '',
        'Incidents': int(total),
        'Frequency': table['Frequency'].sum(),
        'Mean severity': np.average(table['Mean severity'], weights=table['Incidents']),
        'Relative risk': table['Relative risk'].sum(),
        'Risk share': table['Risk share'].sum(),
        'Rank': '—',
    }
    table = pd.concat([table, pd.DataFrame([total_row])], ignore_index=True)

    # 列顺序与保留小数
    out_cols = ['Scenario', 'Incidents', 'Frequency', 'Mean severity',
                'Relative risk', 'Risk share', 'Rank']
    out = table[out_cols].copy()
    out['Frequency'] = (out['Frequency'] * 100).round(1).astype(str) + '%'
    out['Mean severity'] = out['Mean severity'].round(3)
    out['Relative risk'] = out['Relative risk'].round(3)
    out['Risk share'] = (out['Risk share'] * 100).round(1).astype(str) + '%'
    return out


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'config.yaml')
    cfg = load_config(config_path)

    # 优先读取已聚类并带 severity 的结果文件
    input_candidates = [
        os.path.join(base_dir, cfg['data'].get('output_dir', 'output'), 'full_risk_data.csv'),
        os.path.join(base_dir, 'data', 'output.csv'),
    ]
    src = next((p for p in input_candidates if os.path.exists(p)), input_candidates[0])
    print(f"Loading clustered data from: {src}")
    df = pd.read_csv(src)

    severity_map = resolve_severity_mapping(cfg['risk'])
    out = build_cluster_risk_table(df, severity_map)

    out_path = os.path.join(base_dir, cfg['data'].get('output_dir', 'output'),
                            'cluster_risk_table.csv')
    out.to_csv(out_path, index=False)
    print(f"Cluster risk table saved to: {out_path}")
    print(out.to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
