from kmodes.kmodes import KModes
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from stepmix.stepmix import StepMix
from pyclustering.cluster.rock import rock
import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns

class ClusteringModel:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
            self.config = full_config['clustering']
            self.output_dir = full_config['data'].get('output_dir', 'output')
        self.model = None

    def fit_predict(self, df, method=None, n_clusters=None):
        """"""
        method = method or self.config.get('method', 'kmodes')
        n_clusters = n_clusters or self.config['n_clusters']
        
        
        X = df.copy()
        
        if method == 'kmodes':
            
            X_cat = X.astype(str)
            self.model = KModes(
                n_clusters=n_clusters,
                init=self.config['init'],
                n_init=self.config['n_init'],
                max_iter=self.config['max_iter'],
                random_state=self.config['random_state']
            )
            labels = self.model.fit_predict(X_cat)
            
        elif method == 'agglomerative':
            X_encoded = pd.get_dummies(X)
            self.model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=self.config.get('linkage', 'complete')
            )
            labels = self.model.fit_predict(X_encoded)

        elif method == 'lca':
            
            X_encoded = X.astype(str).apply(lambda x: pd.factorize(x, sort=True)[0])
            self.model = StepMix(
                n_components=n_clusters,
                measurement="categorical",
                random_state=self.config['random_state'],
                max_iter=self.config['max_iter'],
                n_init=self.config['n_init'],
                verbose=0
            )
            self.model.fit(X_encoded)
            labels = self.model.predict(X_encoded)

        elif method == 'rock':
            
            
            X_encoded = pd.get_dummies(X).values.tolist()
            
            radius = self.config.get('rock_radius', 0.5)
            self.model = rock(X_encoded, radius, n_clusters)
            self.model.process()
            clusters = self.model.get_clusters()
            
            
            labels = np.zeros(len(X_encoded), dtype=int)
            for cluster_id, idx_list in enumerate(clusters):
                for idx in idx_list:
                    labels[idx] = cluster_id
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
            
        return labels

    def run_robustness_grid_search(self, df, k_range=range(2, 11), methods=['kmodes', 'agglomerative', 'lca', 'rock']):
        """
        K
        """
        print(f"\n{'='*50}\nStarting Robustness Grid Search\n{'='*50}")
        grid_results = []
        df_encoded = pd.get_dummies(df)
        
        for k in k_range:
            for m in methods:
                try:
                    labels = self.fit_predict(df, method=m, n_clusters=k)
                    
                    actual_k = len(np.unique(labels))
                    
                    sil = silhouette_score(df_encoded, labels)
                    db = davies_bouldin_score(df_encoded, labels)
                    
                    grid_results.append({
                        'Algorithm': m,
                        'K': k,
                        'Silhouette Score': round(sil, 3),
                        'DBI': round(db, 3),
                        'Actual Clusters': actual_k
                    })
                except Exception as e:
                    print(f"Error grid search {m} with k={k}: {e}")
                    
        results_df = pd.DataFrame(grid_results)
        
        
        results_df.to_csv(os.path.join(self.output_dir, 'robustness_grid_search.csv'), index=False)
        
        
        pivot_sil = results_df.pivot(index='Algorithm', columns='K', values='Silhouette Score')
        pivot_dbi = results_df.pivot(index='Algorithm', columns='K', values='DBI')
        
        print("\nSilhouette Score Matrix (Higher is better):")
        print(pivot_sil)
        print("\nDavies-Bouldin Index Matrix (Lower is better):")
        print(pivot_dbi)
        
        return results_df

    def calculate_stability(self, df, method, k, n_bootstrap=10):
        """
         (ARI )
        """
        print(f"\nTesting stability for {method} with k={k} ({n_bootstrap} bootstrap iterations)...")
        
        df_with_idx = df.copy()
        df_with_idx['_original_index'] = range(len(df))
        
        base_labels = self.fit_predict(df, method=method, n_clusters=k)
        ari_scores = []
        
        for i in range(n_bootstrap):
            
            sample_df = df_with_idx.sample(frac=0.9, random_state=i)
            sample_indices_in_original = sample_df['_original_index'].values
            
            
            X_sample = sample_df.drop(columns=['_original_index'])
            
            try:
                sample_labels = self.fit_predict(X_sample, method=method, n_clusters=k)
                
                
                original_labels_subset = base_labels[sample_indices_in_original]
                
                ari = adjusted_rand_score(original_labels_subset, sample_labels)
                ari_scores.append(ari)
            except Exception as e:
                print(f"Stability test iteration {i} failed: {e}")
                
        avg_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)
        print(f"Stability Result: Mean ARI = {avg_ari:.3f} (±{std_ari:.3f})")
        
        
        stability_report = pd.DataFrame({
            'Method': [method],
            'K': [k],
            'Mean ARI': [round(avg_ari, 3)],
            'Std ARI': [round(std_ari, 3)],
            'Iterations': [n_bootstrap]
        })
        stability_report.to_csv(os.path.join(self.output_dir, f'stability_{method}_k{k}.csv'), index=False)
        
        return avg_ari

    def compare_algorithms(self, df, methods=['kmodes', 'agglomerative', 'lca', 'rock']):
        """"""
        results = []
        df_encoded = pd.get_dummies(df)
        
        for m in methods:
            print(f"Running clustering comparison: {m}...")
            try:
                labels = self.fit_predict(df, method=m)
                
                sil = silhouette_score(df_encoded, labels)
                db = davies_bouldin_score(df_encoded, labels)
                
                results.append({
                    'Method': m,
                    'Silhouette Score': round(sil, 3),
                    'Davies-Bouldin Index': round(db, 3),
                    'Clusters': len(np.unique(labels))
                })
            except Exception as e:
                print(f"Error comparing {m}: {e}")
                
        comparison_df = pd.DataFrame(results)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        comparison_df.to_csv(os.path.join(self.output_dir, 'clustering_comparison.csv'), index=False)
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        comparison_df.plot(x='Method', y='Silhouette Score', kind='bar', ax=ax[0], color='skyblue')
        ax[0].set_title("Silhouette Score (Higher is better)")
        comparison_df.plot(x='Method', y='Davies-Bouldin Index', kind='bar', ax=ax[1], color='salmon')
        ax[1].set_title("Davies-Bouldin Index (Lower is better)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "clustering_comparison.png"), dpi=300)
        
        return comparison_df

    def analyze_feature_distribution(self, df, labels, columns):
        """"""
        df_with_labels = df.copy()
        df_with_labels['cluster'] = labels
        
        all_results = []
        for col in columns:
            dist = (
                df_with_labels.groupby('cluster', observed=False)[col]
                .value_counts(normalize=True)
                .rename("proportion")
                .reset_index()
            )
            
            dist = dist.rename(columns={col: 'value'})
            dist['feature'] = col
            all_results.append(dist)
            
        result_df = pd.concat(all_results, ignore_index=True)
        
        
        pivot_df = result_df.pivot_table(
            index=['feature', 'value'], 
            columns='cluster',
            values='proportion',
            fill_value=0
        )
        
        
        pivot_df.columns = [f"Cluster {int(c)+1}" for c in pivot_df.columns]
        
        
        cluster_counts = df_with_labels['cluster'].value_counts().sort_index()
        total_samples = len(df_with_labels)
        cluster_proportions = cluster_counts / total_samples
        
        summary_stats = {
            'counts': cluster_counts,
            'proportions': cluster_proportions
        }
        
        return pivot_df, summary_stats
