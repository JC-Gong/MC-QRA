import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from src.data_processor import DataProcessor
from src.clustering import ClusteringModel
from src.risk_calculator import RiskCalculator
from src.visualization import AcademicPlotter
from src.utils import format_excel_thresholds, calculate_mi_matrix
from src.descriptive_stats import generate_descriptive_statistics


def _canon(v):
    """/ Netica : ,  s ,
    / None, """
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ("", "nan", "none", "n/a", "na"):
        return None
    c = "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")
    return ("s" + c) if (c and c[0].isdigit()) else c

def main():
    config_path = "config/config.yaml"
    
    
    processor = DataProcessor(config_path)
    
    
    output_dir = processor.config['data'].get('output_dir', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    min_year = processor.config['data'].get('min_year', 2010)
    max_year = processor.config['data'].get('max_year', None)
    def year_mask(df):
        m = df['Year'] >= min_year
        if max_year is not None:
            m = m & (df['Year'] <= max_year)
        return m
        
    df_raw = processor.load_data()
    df_clean = processor.clean_data(df_raw)
    
    
    clustering = ClusteringModel(config_path)
    
    
    cat_cols = processor.config['data']['categorical_columns']
    df_for_clustering = df_clean[year_mask(df_clean)][cat_cols].copy()
    
    
    k_range = range(4, 10)
    clustering.run_robustness_grid_search(df_for_clustering, k_range=k_range)
    
    
    method = clustering.config.get('method', 'agglomerative')
    n_clusters = clustering.config.get('n_clusters', 7)
    labels = clustering.fit_predict(df_for_clustering)
    df_clean.loc[year_mask(df_clean), 'cluster'] = labels
    
    
    clustering.calculate_stability(df_for_clustering, method=method, k=n_clusters)
    
    
    pivot_df, summary_stats = clustering.analyze_feature_distribution(df_for_clustering, labels, cat_cols)
    
    
    
    feature_levels = pivot_df.index.get_level_values(0)

    
    dominant_mask = pivot_df.groupby(feature_levels, group_keys=False).apply(lambda g: g == g.max())

    
    cross_cluster_max_mask = pivot_df.apply(lambda row: row == row.max(), axis=1)

    
    thresholds_mask = dominant_mask & cross_cluster_max_mask
    
    
    pivot_df_str = pivot_df.map(lambda x: f"{x*100:.1f}%")
    
    
    cluster_counts = summary_stats['counts']
    cluster_proportions = summary_stats['proportions']
    
    
    summary_data = {
        f"Cluster {int(i)+1}": [int(count), f"{prop*100:.1f}%"] 
        for i, (count, prop) in enumerate(zip(cluster_counts, cluster_proportions))
    }
    summary_df = pd.DataFrame(summary_data, index=pd.MultiIndex.from_tuples([('SUMMARY', 'Event Count'), ('SUMMARY', 'Event %')]))
    
    
    final_df = pd.concat([pivot_df_str, summary_df])
    
    output_excel = os.path.join(output_dir, "cluster_analysis.xlsx")
    final_df.to_excel(output_excel)
    
    
    
    format_excel_thresholds(output_excel, pivot_df, thresholds_mask, start_col=3)
    
    
    calculator = RiskCalculator(config_path)
    df_risk = calculator.calculate_risk(df_clean[year_mask(df_clean)].copy())
    df_risk = calculator.get_risk_levels(df_risk)

    
    binning_combinations = calculator.config.get('binning_combinations', [])
    if binning_combinations:
        count_df, prop_df = calculator.compare_binning_schemes(df_risk, binning_combinations)
        prop_pct_df = (prop_df * 100).round(1).astype(str) + '%'
        print("Risk Level Binning Comparison (Counts):")
        print(count_df.to_string())
        print("\nRisk Level Binning Comparison (Proportion):")
        print(prop_pct_df.to_string())
        count_df.to_csv(os.path.join(output_dir, 'risk_binning_counts.csv'))
        prop_df.to_csv(os.path.join(output_dir, 'risk_binning_proportions.csv'))
    
    
    export_config = processor.config['data'].get('export', {})
    year_bins = export_config.get('year_bins', [2010, 2015, 2020, 2023, 2027])
    year_labels = export_config.get('year_labels', ['2010-2014', '2015-2019', '2020-2022', '2023-2026'])
    plotter = AcademicPlotter(config_path)
    plotter.plot_risk_temporal_heatmap(df_risk)
    
    
    geometry = [Point(xy) for xy in zip(df_risk['Lon'], df_risk['Lat'])]
    gdf = gpd.GeoDataFrame(df_risk, geometry=geometry)
    world = gpd.read_file('../shape/ne_110m_admin_0_countries.shp')
    
    plotter.plot_kde_map(gdf, world, (2010, 2025))
    plotter.plot_hotspot_map(gdf, world, (2010, 2025))
    
    plotter.plot_kde_time_evolution(gdf, world, year_bins, year_labels)
    plotter.plot_hotspot_time_evolution(gdf, world, year_bins, year_labels)
    
    
    cat_vars = ['Year', 'Target', 'Asset exploited', 'Cyber threats',
                'Consequence type', 'Intent', 'Origin',  
                'Attack countries', 'Victim countries', 'Consequence severity', 'Risk bin']
    mi_matrix, nmi_matrix = calculate_mi_matrix(df_risk, cat_vars)
    plotter.plot_mi_barh(mi_matrix, target="Risk bin",
                         exclude=["Consequence type", "Consequence severity"])
    
    
    export_config = processor.config['data'].get('export', {})
    binning_method = calculator.config.get('binning_method', 'rank')
    year_bins = export_config.get('year_bins', [2010, 2015, 2020, 2023, 2027])
    year_labels = export_config.get('year_labels', ['2010-2014', '2015-2019', '2020-2022', '2023-2026'])
    default_export_cols = cat_cols + ['Risk bin']
    export_cols = export_config.get('columns', default_export_cols)
    n_splits = export_config.get('n_splits', 5)
    random_state = export_config.get('random_state', 42)

    
    df_full = df_risk.copy()
    if 'Year' in df_full.columns:
        df_full['Year_Group'] = pd.cut(df_full['Year'], bins=year_bins,
                                       labels=year_labels, right=False).astype(str)
    rank_col_name = 'Risk Rank (Unique)' if binning_method == 'rank' else 'Risk Score (Norm)'
    df_full = df_full.rename(columns={
        'risk': 'Risk Score',
        'risk_rank': rank_col_name,
        'Risk bin': 'Risk Level'
    })
    df_full.to_csv(os.path.join(output_dir, 'full_risk_data.csv'), index=False)

    
    generate_descriptive_statistics(
        df_full,
        output_dir,
        year_bins=year_bins,
        year_labels=year_labels,
        min_year=min_year,
        max_year=max_year,
    )

    
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_combos = calculator.config.get('binning_combinations', [])
    selected = set(calculator.config.get('selected_combinations', [c['name'] for c in all_combos]))

    
    bn_cfg = processor.config.get('netica', {})
    export_scenarios = bn_cfg.get('export_scenarios', [
        {"name": "without_intent_origin", "extra_columns": []},
        {"name": "with_intent_origin", "extra_columns": ['Intent', 'Origin']},
    ])

    def build_export(df_src, cols):
        d = df_src[cols].copy()
        if 'Year' in d.columns:
            d['Year'] = pd.cut(d['Year'], bins=year_bins, labels=year_labels, right=False).astype(str)
            d['Year'] = d['Year'].replace('nan', 'Unknown')
        if 'Risk bin' in d.columns:
            d = d.rename(columns={'Risk bin': 'Risk Level'})
        
        
        d = d.rename(columns={c: _canon(c) for c in d.columns})
        for c in d.columns:
            d[c] = d[c].map(lambda v: _canon(v) if pd.notna(v) else v)
        return d

    for scen in export_scenarios:
        scenario_name = scen['name']
        scen_cols = list(export_cols) + list(scen.get('extra_columns', []))
        scen_root = os.path.join(output_dir, scenario_name)
        if not os.path.exists(scen_root):
            os.makedirs(scen_root)
        for combo in all_combos:
            if combo['name'] not in selected:
                continue
            combo_dir = os.path.join(scen_root, combo['name'])
            if not os.path.exists(combo_dir):
                os.makedirs(combo_dir)

            
            df_combo = calculator.get_risk_levels(df_risk.copy(), bins=combo['bins'], labels=combo['labels'])
            df_export = build_export(df_combo, scen_cols)

            
            stratify_target = df_combo.loc[df_export.index, 'Risk bin']
            for i, (tr, te) in enumerate(kf.split(df_export, stratify_target)):
                df_export.iloc[tr].to_csv(os.path.join(combo_dir, f'train_fold_{i+1}.csv'), index=False)
                df_export.iloc[te].to_csv(os.path.join(combo_dir, f'test_fold_{i+1}.csv'), index=False)
            
            df_export.to_csv(os.path.join(combo_dir, 'all_data.csv'), index=False)
            df_export.to_csv(os.path.join(combo_dir, 'risk_results.csv'), index=False)
            print(f"[{scenario_name}] BN test files generated: {combo['name']} -> {combo_dir}")

    
    
    from src.netica_bn import load_netica, validate_combo, load_config
    netica_cfg = load_config(config_path)
    
    
    structure_methods = netica_cfg.get('structure_methods', [netica_cfg.get('structure_method', 'tan')])
    netica, _saved = load_netica(netica_cfg)
    for sm in structure_methods:
        print(f"[Netica CV] structure_method = {sm}")
        sm_cfg = dict(netica_cfg)
        sm_cfg['structure_method'] = sm
        env = netica.Environ()
        try:
            for scen in export_scenarios:
                scenario_name = scen['name']
                scen_root = os.path.join(output_dir, scenario_name)
                for combo in all_combos:
                    if combo['name'] not in selected:
                        continue
                    combo_dir = os.path.join(scen_root, combo['name'])
                    validate_combo(netica, env, combo_dir, sm_cfg)
        finally:
            env.close_netica()

    
    from src.bn_metrics import summarize
    summary_csv = os.path.join(output_dir, "bn_metrics_cv_summary.csv")
    binning = {c['name']: (c.get('bins'), c.get('labels')) for c in all_combos}
    allow = {(scen['name'], c['name']) for scen in export_scenarios
             for c in all_combos if c['name'] in selected}
    summarize(output_dir, dataset="MC-RA", out_csv=summary_csv, allow=allow, binning=binning)
    print(f"BN validation metrics summary written to: {summary_csv}")

    print("MC-RA Analysis Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()
