import os
import pandas as pd


COUNTRY_NAME_MAP = {
    'Russian Federation': 'Russia',
    'Republic Of Korea': 'South Korea',
    'Other': 'Others',
}


DESCRIPTIVE_VARIABLES = [
    'Year',
    'Attack countries',
    'Victim countries',
    'Target',
    'Cyber threats',
    'Consequence severity',
    'Intent',
    'Origin',
    'Asset exploited',
    'Consequence type',
]


def _clean_country(value):
    return COUNTRY_NAME_MAP.get(value, value)


def _round_percentages_to_100(counts, n):
    """(1),  100.0%

    :  1 ,  0.1%
    counts: list of (category, count)
    """
    if n == 0:
        return [(k, c, 0.0) for k, c in counts]
    floors = []  # (category, count, truncated_pct, remainder)
    for k, c in counts:
        raw = c / n * 100.0
        truncated = int(raw * 10) / 10.0
        floors.append((k, c, truncated, raw - truncated))
    total_floor = sum(f for _, _, f, _ in floors)
    deficit = int(round((100.0 - total_floor) * 10))  
    deficit = max(0, min(deficit, len(floors)))
    
    order = sorted(range(len(floors)),
                   key=lambda i: (floors[i][3], floors[i][1]),
                   reverse=True)
    adjust = set(order[:deficit])
    result = []
    for i, (k, c, f, _) in enumerate(floors):
        add = 0.1 if i in adjust else 0.0
        result.append((k, c, round(f + add, 1)))
    return result


def generate_descriptive_statistics(df, output_dir,
                                    year_bins=None, year_labels=None,
                                    min_year=2010, max_year=None):
    """ 2010 

     output_dir:
        - descriptive_statistics.tsv :  (Variable \\t Cat (pct%) ...)
        - descriptive_statistics.md  :  (Variable / Category / Count / Percentage)

    Year  year_bins / year_labels , 
    """
    if year_bins is None:
        year_bins = [2010, 2015, 2020, 2023, 2027]
    if year_labels is None:
        year_labels = ['2010-2014', '2015-2019', '2020-2022', '2023-2026']

    
    mask = df['Year'] >= min_year
    if max_year is not None:
        mask = mask & (df['Year'] <= max_year)
    df_sub = df[mask].copy()
    n = len(df_sub)
    if n == 0:
        print("Warning: no data after 2010, skipping descriptive statistics.")
        return

    
    df_sub['Year_Group'] = pd.cut(df_sub['Year'], bins=year_bins,
                                  labels=year_labels, right=False).astype(str)
    df_sub['Year_Group'] = df_sub['Year_Group'].replace('nan', 'Unknown')

    
    raw_counts = {}
    for var in DESCRIPTIVE_VARIABLES:
        col = 'Year_Group' if var == 'Year' else var
        mapper = _clean_country if var in ('Attack countries', 'Victim countries') else None
        series = df_sub[col].astype(str).map(mapper) if mapper else df_sub[col].astype(str)
        vc = series.value_counts()
        raw_counts[var] = [(k, int(v)) for k, v in vc.items()]

    
    counts = {}
    for var in DESCRIPTIVE_VARIABLES:
        counts[var] = _round_percentages_to_100(raw_counts[var], n)

    
    tsv_lines = []
    for var in DESCRIPTIVE_VARIABLES:
        parts = [f"{k} ({p:.1f}%)" for k, _, p in counts[var]]
        tsv_lines.append(f"{var}\t" + "\t".join(parts))
    tsv_path = os.path.join(output_dir, 'descriptive_statistics.tsv')
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(tsv_lines) + "\n")

    
    md_lines = [
        "| Variable | Category | Count | Percentage |",
        "|---|---|---:|---:|",
    ]
    for var in DESCRIPTIVE_VARIABLES:
        first = True
        for k, c, p in counts[var]:
            vname = var if first else ""
            first = False
            md_lines.append(f"| {vname} | {k} | {c} | {p:.1f}% |")
    md_path = os.path.join(output_dir, 'descriptive_statistics.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines) + "\n")

    return tsv_path, md_path
