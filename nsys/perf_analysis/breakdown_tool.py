import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn for clustering
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from k_means_constrained import KMeansConstrained
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Clustering analysis will be disabled.")

# Set up font support for plots
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

KEEP_SRC_COLUMNS = ["kernel_id", "Name", "Start Timestamp", "Context", "Duration Value"]
MIN_CLUSTER_SIZE = 2  # Minimum number of samples per cluster

def get_output_filename_base(csv_path):
    """Extract base filename from CSV path for consistent output naming"""
    # Get filename without path and extension
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    # Convert to lowercase and replace any problematic characters
    base_name = base_name.lower().replace(' ', '_')
    return base_name

def determine_kernel_type(row):
    """Determine kernel type with focus on fMHA, GEMM, and NCCL classification"""
    kernel_name = row["Name"]
    kernel_name_lower = kernel_name.lower()
    
    if any(keyword in kernel_name_lower for keyword in ['fmha', 'flash', 'attention']):
        return "fMHA"
    elif any(keyword in kernel_name_lower for keyword in ['gemm', 'nvjet']):
        return "GEMM"
    elif any(keyword in kernel_name_lower for keyword in ['nccl']):
        return "NCCL"
    else:
        return "Others"

def analyze_fmha_subtypes(df_fmha):
    """Analyze fMHA kernel subtypes"""
    subtypes = {}
    
    for _, row in df_fmha.iterrows():
        kernel_name = row["Name"].lower()
        
        # Classify based on kernel name features
        if 'forward' in kernel_name or 'fwd' in kernel_name:
            subtype = "fMHA_Forward"
        elif 'backward' in kernel_name or 'bwd' in kernel_name:
            subtype = "fMHA_Backward"
        elif 'flash_attn' in kernel_name:
            if 'v2' in kernel_name:
                subtype = "FlashAttention_v2"
            elif 'v1' in kernel_name:
                subtype = "FlashAttention_v1"
            else:
                subtype = "FlashAttention"
        elif 'scaled_dot_product' in kernel_name:
            subtype = "ScaledDotProduct"
        elif 'multi_head' in kernel_name:
            subtype = "MultiHead"
        else:
            subtype = "fMHA_Other"
        
        if subtype not in subtypes:
            subtypes[subtype] = []
        subtypes[subtype].append(row)
    
    return subtypes

def analyze_gemm_subtypes(df_gemm):
    """Analyze GEMM kernel subtypes"""
    subtypes = {}
    
    for _, row in df_gemm.iterrows():
        kernel_name = row["Name"].lower()
        
        # Classify based on GEMM variants
        if 'sgemm' in kernel_name:
            subtype = "SGEMM_FP32"
        elif 'hgemm' in kernel_name or 'half' in kernel_name:
            subtype = "HGEMM_FP16"
        elif 'igemm' in kernel_name:
            subtype = "IGEMM_INT"
        elif 'batched' in kernel_name or 'batch' in kernel_name:
            subtype = "BatchedGEMM"
        elif 'strided' in kernel_name:
            subtype = "StridedGEMM"
        elif 'bias' in kernel_name:
            subtype = "GEMM_Bias"
        elif 'relu' in kernel_name or 'gelu' in kernel_name:
            subtype = "GEMM_Activation"
        else:
            subtype = "GEMM_Basic"
        
        if subtype not in subtypes:
            subtypes[subtype] = []
        subtypes[subtype].append(row)
    
    return subtypes

def analyze_nccl_subtypes(df_nccl):
    """Analyze NCCL kernel subtypes"""
    subtypes = {}
    
    for _, row in df_nccl.iterrows():
        kernel_name = row["Name"].lower()
        
        # Classify based on NCCL operation types
        if 'sendrecv' in kernel_name:
            subtype = "SendRecv"
        elif 'allgather' in kernel_name:
            subtype = "AllGather"
        else:
            subtype = "NCCL_Other"

        if subtype not in subtypes:
            subtypes[subtype] = []
        subtypes[subtype].append(row)
    
    return subtypes

def perform_clustering_analysis(df_kernel, kernel_type='fMHA', min_cluster_size=MIN_CLUSTER_SIZE):
    """Perform clustering analysis on kernel data to identify different performance clusters"""
    if not SKLEARN_AVAILABLE:
        return None, None, None
        
    if len(df_kernel) < 4:  # Need at least 4 samples for meaningful clustering
        return None, None, None
    
    # Prepare features for clustering
    features = []
    
    # Duration is the primary feature
    durations = df_kernel['Duration'].values.reshape(-1, 1)
    
    # Primary feature is duration
    features.insert(0, durations)
    X = np.hstack(features)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Different clustering strategies for different kernel types
    if kernel_type == 'GEMM':
        # GEMM-specific clustering: more aggressive clustering, expecting larger variance
        max_clusters = min(8, len(df_kernel)//3)  # Allow more clusters for GEMM
        cluster_range = range(4, max_clusters)
        
        # Use different DBSCAN parameters for GEMM
        dbscan_eps = 0.3  # Tighter epsilon for GEMM
        dbscan_min_samples = max(4, len(df_kernel)//15)  # Stricter min samples
    elif kernel_type == 'NCCL':
        # NCCL-specific clustering: communication operations can have high variance
        max_clusters = min(6, len(df_kernel)//3)  # Moderate clustering for NCCL
        cluster_range = range(2, max_clusters)
        
        # Use moderate DBSCAN parameters for NCCL
        dbscan_eps = 0.4  # Slightly looser epsilon for NCCL
        dbscan_min_samples = max(3, len(df_kernel)//12)  # Moderate min samples
    else:
        # fMHA-specific clustering: original parameters
        max_clusters = min(6, len(df_kernel)//2)
        cluster_range = range(2, max_clusters)
        
        dbscan_eps = 0.3
        dbscan_min_samples = max(4, len(df_kernel)//10)
    
    # Try different numbers of clusters and find optimal
    best_score = -1
    best_n_clusters = 2
    best_labels = None
    
    for n_clusters in cluster_range:
        try:
            # kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            kmeans = KMeansConstrained(n_clusters=n_clusters, 
                                       size_min=MIN_CLUSTER_SIZE, 
                                       size_max=len(df_kernel), 
                                       random_state=0,
                                       n_init=10,
                                       max_iter=500,
                                       n_jobs=-2)
            labels = kmeans.fit_predict(X_scaled)
            
            # Calculate silhouette score
            score = silhouette_score(X_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels
        except:
            continue
    print(f"[INFO] KMeansConstrained: {best_n_clusters} clusters, score={best_score:.3f}")

    # Also try DBSCAN as an alternative
    try:
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # Check if DBSCAN found meaningful clusters
        n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        if n_dbscan_clusters >= 2:
            dbscan_score = silhouette_score(X_scaled, dbscan_labels) if n_dbscan_clusters > 1 else -1
            if dbscan_score > best_score:
                best_score = dbscan_score
                best_labels = dbscan_labels
                best_n_clusters = n_dbscan_clusters
            print(f"[INFO] DBSCAN: {n_dbscan_clusters} clusters, score={dbscan_score:.3f}")
    except:
        pass
    
    if best_labels is None:
        return None, None, None
    
    return best_labels, best_n_clusters, best_score

def merge_similar_clusters(df_fmha, cluster_labels, threshold_percent=0.1):
    """Merge clusters with similar mean durations (within threshold_percent)"""
    if cluster_labels is None:
        return cluster_labels
    
    unique_clusters = sorted(set(cluster_labels))
    if len(unique_clusters) <= 1:
        return cluster_labels
    
    # Calculate mean duration for each cluster
    cluster_means = {}
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise points
            continue
        cluster_mask = cluster_labels == cluster_id
        cluster_data = df_fmha[cluster_mask]
        cluster_means[cluster_id] = cluster_data['Duration'].mean()
    
    # Find clusters to merge
    merged_labels = cluster_labels.copy()
    cluster_mapping = {}  # old_id -> new_id
    
    # Sort clusters by mean duration
    sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])
    
    for i, (cluster_id, mean_duration) in enumerate(sorted_clusters):
        if cluster_id in cluster_mapping:
            continue
            
        # Find similar clusters to merge
        clusters_to_merge = [cluster_id]
        
        for j, (other_cluster_id, other_mean) in enumerate(sorted_clusters):
            if i != j and other_cluster_id not in cluster_mapping:
                # Calculate percentage difference
                diff_percent = abs(mean_duration - other_mean) / max(mean_duration, other_mean)
                
                if diff_percent <= threshold_percent:
                    clusters_to_merge.append(other_cluster_id)
        
        # Assign new cluster ID (use the lowest original ID)
        new_cluster_id = min(clusters_to_merge)
        for old_cluster_id in clusters_to_merge:
            cluster_mapping[old_cluster_id] = new_cluster_id
    
    # Apply cluster mapping
    for old_id, new_id in cluster_mapping.items():
        merged_labels[cluster_labels == old_id] = new_id
    
    # Renumber clusters to be consecutive starting from 0
    unique_merged = sorted(set(merged_labels))
    if -1 in unique_merged:
        unique_merged.remove(-1)  # Keep noise points as -1
    
    final_labels = merged_labels.copy()
    for i, cluster_id in enumerate(unique_merged):
        final_labels[merged_labels == cluster_id] = i
    
    # Keep noise points as -1
    if -1 in set(merged_labels):
        final_labels[merged_labels == -1] = -1
    
    n_final_clusters = len(unique_merged)
    
    return final_labels, n_final_clusters

def validate_clustering_quality(df_kernel, cluster_labels, kernel_type):
    """Validate clustering quality and reject poor clusters (except for GEMM which always proceeds)"""
    if cluster_labels is None:
        return False
    
    unique_clusters = sorted(set(cluster_labels))
    if len(unique_clusters) <= 1:
        return False
    
    # Check cluster quality
    if kernel_type == 'GEMM':
        print(f"📊 GEMM Clustering Statistics ({len(unique_clusters)} clusters):")
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_data = df_kernel[cluster_mask]['Duration']
            
            if len(cluster_data) < 2:
                continue
                
            cluster_std = cluster_data.std()
            cluster_mean = cluster_data.mean()
            cluster_count = len(cluster_data)
            cv = cluster_std / cluster_mean if cluster_mean > 0 else float('inf')
            
            # For GEMM, just show information without rejecting
            status = ""
            if cv > 1.5:
                status = f"(High CV: {cv:.2f})"
            elif cluster_std > 100:
                status = f"(High Std: {cluster_std:.2f})"
            else:
                status = "(Good)"
                
            print(f"  Cluster {cluster_id}: count={cluster_count}, mean={cluster_mean:.3f}, std={cluster_std:.3f} {status}")
        
        print("✅ GEMM clustering proceeding (validation bypassed)")
        return True  # Always return True for GEMM
    elif kernel_type == 'NCCL':
        print(f"📊 NCCL Clustering Statistics ({len(unique_clusters)} clusters):")
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_data = df_kernel[cluster_mask]['Duration']
            
            if len(cluster_data) < 2:
                continue
                
            cluster_std = cluster_data.std()
            cluster_mean = cluster_data.mean()
            cluster_count = len(cluster_data)
            cv = cluster_std / cluster_mean if cluster_mean > 0 else float('inf')
            
            # For NCCL, show information and apply moderate validation
            status = ""
            if cv > 2.0:
                status = f"(High CV: {cv:.2f})"
            elif cluster_std > 200:
                status = f"(High Std: {cluster_std:.2f})"
            else:
                status = "(Good)"
                
            print(f"  Cluster {cluster_id}: count={cluster_count}, mean={cluster_mean:.3f}, std={cluster_std:.3f} {status}")
        
        print("✅ NCCL clustering proceeding (moderate validation applied)")
        return True  # Return True for NCCL with moderate validation
    else:
        # For other kernel types, apply strict validation
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_data = df_kernel[cluster_mask]['Duration']
            
            if len(cluster_data) < 2:
                continue
                
            cluster_std = cluster_data.std()
            cluster_mean = cluster_data.mean()
            cv = cluster_std / cluster_mean if cluster_mean > 0 else float('inf')
            
            # For non-GEMM kernels, apply validation
            if cv > 1.5:
                print(f"❌ Cluster {cluster_id} rejected: high variability (CV={cv:.2f} > 1.5)")
                return False
                
            if cluster_std > 100:
                print(f"❌ Cluster {cluster_id} rejected: high standard deviation ({cluster_std:.2f} > 100)")
                return False
        
        print("✅ All clusters passed quality validation")
        return True





def analyze_cluster_performance(df_cluster, cluster_id, kernel_type):
    """Analyze performance patterns for a specific cluster"""
    duration_stats = df_cluster['Duration'].describe()
    
    # Calculate outliers within cluster
    Q1 = duration_stats['25%']
    Q3 = duration_stats['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_cluster[(df_cluster['Duration'] < lower_bound) | 
                         (df_cluster['Duration'] > upper_bound)]
    
    # Performance tier analysis within cluster
    p50 = duration_stats['50%']
    p90 = df_cluster['Duration'].quantile(0.9)
    p99 = df_cluster['Duration'].quantile(0.99)
    
    performance_tiers = {
        'Fast': len(df_cluster[df_cluster['Duration'] <= p50]),
        'Medium': len(df_cluster[(df_cluster['Duration'] > p50) & (df_cluster['Duration'] <= p90)]),
        'Slow': len(df_cluster[(df_cluster['Duration'] > p90) & (df_cluster['Duration'] <= p99)]),
        'Very Slow': len(df_cluster[df_cluster['Duration'] > p99])
    }
    
    # Calculate sum for stats
    total_duration = df_cluster['Duration'].sum()
    
    return {
        'cluster_id': cluster_id,
        'stats': duration_stats,
        'total_duration': total_duration,
        'outliers': outliers,
        'performance_tiers': performance_tiers,
        'percentiles': {
            'p50': p50,
            'p90': p90,
            'p99': p99
        }
    }



def generate_unified_cluster_report(df_kernel, cluster_labels, n_clusters, silhouette_score_val, subtypes, kernel_type, output_filename):
    """Generate unified cluster analysis report combining all clusters"""
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"# {kernel_type} Deep Analysis Report with Clustering\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        f.write("## 1. Overall Statistics\n\n")
        total_duration = df_kernel['Duration'].sum()
        overall_stats = df_kernel['Duration'].describe()
        f.write(f"- **Total Call Count**: {int(overall_stats['count'])}\n")
        f.write(f"- **Total Execution Time**: {total_duration:.3f}\n")
        f.write(f"- **Average Execution Time**: {overall_stats['mean']:.3f}\n")
        f.write(f"- **Median Execution Time**: {overall_stats['50%']:.3f}\n")
        f.write(f"- **Standard Deviation**: {overall_stats['std']:.3f}\n\n")
        
        # Clustering overview
        f.write("## 2. Clustering Analysis Overview\n\n")
        f.write(f"- **Number of Clusters Found**: {n_clusters}\n")
        f.write(f"- **Silhouette Score**: {silhouette_score_val:.3f}\n")
        f.write(f"- **Clustering Quality**: ")
        
        if silhouette_score_val > 0.7:
            f.write("Excellent clustering\n")
        elif silhouette_score_val > 0.5:
            f.write("Good clustering\n")
        elif silhouette_score_val > 0.3:
            f.write("Moderate clustering\n")
        else:
            f.write("Weak clustering\n")
        
        f.write("\n")
        
        # Cluster distribution summary
        f.write("## 3. Cluster Distribution Summary\n\n")
        f.write("| Cluster ID | Data Points | Percentage | Mean Duration | Std Duration | Min | Max |\n")
        f.write("|------------|-------------|------------|---------------|--------------|-----|-----|\n")
        
        unique_clusters = sorted(set(cluster_labels))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # Remove noise points
        
        total_points = len(df_kernel)
        cluster_analyses = {}
        
        # Create cluster renaming map based on mean duration (same as visualization)
        cluster_means = []
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = df_kernel[cluster_mask]
            cluster_means.append((cluster_id, cluster_data['Duration'].mean()))
        
        # Sort by mean duration and create renaming map
        cluster_means.sort(key=lambda x: x[1])  # Sort by mean duration
        cluster_rename_map = {}
        for new_id, (old_id, _) in enumerate(cluster_means):
            cluster_rename_map[old_id] = new_id
        
        # Use sorted cluster order for table
        for i, (old_cluster_id, _) in enumerate(cluster_means):
            cluster_mask = cluster_labels == old_cluster_id
            cluster_data = df_kernel[cluster_mask]
            new_cluster_id = cluster_rename_map[old_cluster_id]
            
            count = len(cluster_data)
            percentage = (count / total_points) * 100
            mean_duration = cluster_data['Duration'].mean()
            std_duration = cluster_data['Duration'].std()
            min_duration = cluster_data['Duration'].min()
            max_duration = cluster_data['Duration'].max()
            
            # Store cluster analysis for detailed sections (using old_cluster_id for lookup)
            cluster_analyses[new_cluster_id] = analyze_cluster_performance(cluster_data, new_cluster_id, kernel_type)
            
            f.write(f"| {new_cluster_id} | {count} | {percentage:.1f}% | {mean_duration:.3f} | {std_duration:.3f} | {min_duration:.3f} | {max_duration:.3f} |\n")
        
        f.write("\n")
        
        # Inter-cluster analysis
        f.write("## 4. Inter-Cluster Performance Analysis\n\n")
        
        # Use the already calculated cluster_means and cluster_rename_map
        cluster_mean_values = [mean for _, mean in cluster_means]
        
        if len(cluster_mean_values) >= 2:
            min_mean = min(cluster_mean_values)
            max_mean = max(cluster_mean_values)
            performance_ratio = max_mean / min_mean if min_mean > 0 else float('inf')
            
            f.write(f"- **Performance Range**: {min_mean:.3f} to {max_mean:.3f}\n")
            f.write(f"- **Performance Ratio**: {performance_ratio:.2f}x (slowest/fastest cluster)\n")
            
            # Find fastest and slowest clusters (already sorted by mean)
            fastest_cluster = 0  # First cluster in sorted order (smallest mean)
            slowest_cluster = len(cluster_means) - 1  # Last cluster in sorted order (largest mean)
            
            f.write(f"- **Fastest Cluster**: Cluster {fastest_cluster} (mean: {min_mean:.3f})\n")
            f.write(f"- **Slowest Cluster**: Cluster {slowest_cluster} (mean: {max_mean:.3f})\n\n")
            
            if performance_ratio > 5.0:
                f.write("### ⚠️ High Performance Variance Warning\n")
                f.write(f"- Performance ratio of {performance_ratio:.1f}x indicates significant variance\n")
                f.write("- This suggests potential for substantial optimization gains\n")
                f.write("- Priority should be given to understanding the root cause\n\n")
        
        # Detailed analysis for each cluster
        f.write("## 5. Detailed Cluster Analysis\n\n")
        
        # Use renamed cluster order
        for new_cluster_id in range(len(cluster_means)):
            if new_cluster_id not in cluster_analyses:
                continue
                
            cluster_analysis = cluster_analyses[new_cluster_id]
            
            f.write(f"### Cluster {new_cluster_id}\n\n")
            
            # Basic statistics for this cluster
            stats = cluster_analysis['stats']
            total_cluster_duration = cluster_analysis['total_duration']
            f.write(f"- **Call Count**: {int(stats['count'])}\n")
            f.write(f"- **Total Execution Time**: {total_cluster_duration:.3f}\n")
            f.write(f"- **Average Execution Time**: {stats['mean']:.3f}\n")
            f.write(f"- **Median Execution Time**: {stats['50%']:.3f}\n")
            f.write(f"- **Standard Deviation**: {stats['std']:.3f}\n")
            
            # Performance characteristics
            cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0
            f.write(f"- **Coefficient of Variation**: {cv:.3f}\n")
            
            # Cluster type classification
            if stats['mean'] < 1.0:
                cluster_type = "High-Performance"
                f.write("- **Cluster Type**: High-Performance (fast execution)\n")
            elif stats['mean'] > 10.0:
                cluster_type = "Heavy-Computation"
                f.write("- **Cluster Type**: Heavy-Computation (slow execution)\n")
            else:
                cluster_type = "Standard-Performance"
                f.write("- **Cluster Type**: Standard-Performance (typical execution)\n")
            
            # Performance tiers within cluster
            tiers = cluster_analysis['performance_tiers']
            total_cluster_calls = sum(tiers.values())
            
            f.write("- **Performance Tiers within Cluster**:\n")
            for tier, count in tiers.items():
                percentage = (count / total_cluster_calls * 100) if total_cluster_calls > 0 else 0
                f.write(f"  - {tier}: {count} calls ({percentage:.1f}%)\n")
            
            # Outliers within cluster
            outliers = cluster_analysis['outliers']
            f.write(f"- **Outliers in Cluster**: {len(outliers)} ({(len(outliers)/len(cluster_data)*100):.1f}%)\n")
            
            f.write("\n")
        
        # Subtype analysis across clusters
        if subtypes:
            f.write("## 6. Subtype Analysis Across Clusters\n\n")
            
            # Create a more direct mapping approach
            # Reset index to ensure continuous indexing
            df_kernel_reset = df_kernel.reset_index(drop=True)
            
            # Analyze each cluster separately for subtype distribution
            unique_clusters = sorted(set(cluster_labels))
            
            # Create cluster renaming map for subtypes (same logic as above)
            unique_clusters_sub = sorted(set(cluster_labels))
            if -1 in unique_clusters_sub:
                unique_clusters_sub.remove(-1)
            
            cluster_means_sub = []
            for cluster_id in unique_clusters_sub:
                cluster_mask = cluster_labels == cluster_id
                cluster_data = df_kernel_reset[cluster_mask]
                cluster_means_sub.append((cluster_id, cluster_data['Duration'].mean()))
            
            cluster_means_sub.sort(key=lambda x: x[1])
            cluster_rename_map_sub = {}
            for new_id, (old_id, _) in enumerate(cluster_means_sub):
                cluster_rename_map_sub[old_id] = new_id
            
            # First, get subtypes for each cluster directly
            cluster_subtypes = {}
            for old_cluster_id, _ in cluster_means_sub:
                cluster_mask = cluster_labels == old_cluster_id
                df_cluster = df_kernel_reset[cluster_mask]
                new_cluster_id = cluster_rename_map_sub[old_cluster_id]
                
                # Analyze subtypes for this specific cluster based on kernel type
                if kernel_type == 'fMHA':
                    cluster_subtypes[new_cluster_id] = analyze_fmha_subtypes(df_cluster)
                elif kernel_type == 'GEMM':
                    cluster_subtypes[new_cluster_id] = analyze_gemm_subtypes(df_cluster)
                else:
                    cluster_subtypes[new_cluster_id] = {}
            
            # Now create subtype cross-cluster analysis
            all_subtype_names = set()
            for cluster_id, cluster_sub in cluster_subtypes.items():
                all_subtype_names.update(cluster_sub.keys())
            
            for subtype_name in sorted(all_subtype_names):
                f.write(f"### {subtype_name}\n\n")
                
                subtype_cluster_dist = {}
                total_subtype_count = 0
                
                for cluster_id, cluster_sub in cluster_subtypes.items():
                    if subtype_name in cluster_sub:
                        count = len(cluster_sub[subtype_name])
                        subtype_cluster_dist[cluster_id] = count
                        total_subtype_count += count
                
                if subtype_cluster_dist and total_subtype_count > 0:
                    f.write("| Cluster | Count | Percentage of Subtype |\n")
                    f.write("|---------|-------|------------------------|\n")
                    
                    for cluster_id in sorted(subtype_cluster_dist.keys()):
                        count = subtype_cluster_dist[cluster_id]
                        percentage = (count / total_subtype_count * 100)
                        f.write(f"| {cluster_id} | {count} | {percentage:.1f}% |\n")
                else:
                    f.write("*No cluster distribution data available for this subtype.*\n")
                
                f.write("\n")
        
        # Clustering insights and recommendations
        f.write("## 7. Clustering Insights and Optimization Recommendations\n\n")
        
        f.write("### Clustering Insights:\n")
        if n_clusters == 2:
            f.write("- **Two-Cluster Pattern**: The data shows a clear bimodal distribution\n")
            f.write("- This suggests two distinct execution modes or configurations\n")
            f.write("- Consider investigating the root cause of this bifurcation\n")
        elif n_clusters == 3:
            f.write("- **Three-Cluster Pattern**: The data shows a trimodal distribution\n")
            f.write("- This might indicate different input sizes, batch sizes, or execution paths\n")
            f.write("- Consider analyzing correlation with input parameters\n")
        elif n_clusters >= 4:
            f.write("- **Multi-Cluster Pattern**: The data shows complex clustering patterns\n")
            f.write("- This suggests multiple factors affecting performance\n")
            f.write("- Consider deeper investigation into execution contexts\n")
        
        f.write("\n### General Optimization Recommendations:\n")
        f.write("- Focus optimization efforts on the largest or slowest clusters\n")
        f.write("- Investigate why different clusters exist - this could reveal optimization opportunities\n")
        f.write("- Consider if faster clusters' configurations can be applied to slower ones\n")
        f.write("- Monitor cluster distribution over time to detect performance regressions\n")
        
        # Specific recommendations based on kernel type
        if kernel_type == "fMHA":
            f.write("\n### fMHA-Specific Recommendations:\n")
            f.write("- Different clusters may indicate different sequence lengths or attention patterns\n")
            f.write("- Consider optimizing kernel selection based on input characteristics\n")
            f.write("- Investigate whether clusters correlate with different FlashAttention variants\n")
            f.write("- Consider adaptive kernel dispatch based on cluster characteristics\n")
        elif kernel_type == "GEMM":
            f.write("\n### GEMM-Specific Recommendations:\n")
            f.write("- Different clusters may indicate different matrix sizes or data types\n")
            f.write("- Consider using Tensor Core optimized kernels for appropriate clusters\n")
            f.write("- Investigate whether clusters correlate with batch sizes or matrix shapes\n")
            f.write("- Consider mixed precision strategies based on cluster characteristics\n")
            f.write("- Evaluate kernel selection based on matrix dimensions and memory layout\n")
        elif kernel_type == "NCCL":
            f.write("\n### NCCL-Specific Recommendations:\n")
            f.write("- Different clusters may indicate different communication patterns or data sizes\n")
            f.write("- Investigate whether clusters correlate with different NCCL operations (AllReduce, AllGather, etc.)\n")
            f.write("- Consider optimizing network topology and bandwidth utilization\n")
            f.write("- Analyze cluster patterns for potential communication bottlenecks\n")
            f.write("- Consider using NCCL tuning parameters based on cluster characteristics\n")
            f.write("- Evaluate if clusters correspond to different GPU counts or network configurations\n")
        
        f.write("\n")
        
        # Time distribution characteristics
        f.write("## 8. Overall Time Distribution Characteristics\n\n")
        skewness = df_kernel['Duration'].skew()
        kurtosis = df_kernel['Duration'].kurtosis()
        
        f.write(f"- **Skewness**: {skewness:.3f} ")
        if skewness > 1:
            f.write("(right-skewed, few high-latency calls exist)\n")
        elif skewness < -1:
            f.write("(left-skewed, most calls have high latency)\n")
        else:
            f.write("(approximately normal distribution)\n")
        
        f.write(f"- **Kurtosis**: {kurtosis:.3f} ")
        if kurtosis > 3:
            f.write("(peaked distribution, high concentration)\n")
        elif kurtosis < 3:
            f.write("(flat distribution, high dispersion)\n")
        else:
            f.write("(normal distribution)\n")
        
        f.write("\n")



def create_performance_plots(df_target, kernel_type, output_dir):
    """Create performance analysis charts"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{kernel_type} Kernel Performance Deep Analysis', fontsize=16, fontweight='bold')
    
    # 1. Duration distribution histogram
    axes[0, 0].hist(df_target['Duration'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title(f'{kernel_type} Duration Distribution')
    axes[0, 0].set_xlabel('Duration')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Duration boxplot
    axes[0, 1].boxplot(df_target['Duration'], patch_artist=True, 
                       boxprops=dict(facecolor='lightcoral', alpha=0.7))
    axes[0, 1].set_title(f'{kernel_type} Duration Boxplot')
    axes[0, 1].set_ylabel('Duration')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Timeline analysis
    if 'Start Timestamp' in df_target.columns:
        # Convert timestamp to relative time
        start_times = pd.to_numeric(df_target['Start Timestamp'], errors='coerce')
        start_times = start_times - start_times.min()  # Relative time
        axes[1, 0].scatter(start_times, df_target['Duration'], alpha=0.6, color='green', s=20)
        axes[1, 0].set_title(f'{kernel_type} Timeline Analysis')
        axes[1, 0].set_xlabel('Relative Time')
        axes[1, 0].set_ylabel('Duration')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution function
    sorted_durations = np.sort(df_target['Duration'])
    cumulative_prob = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    axes[1, 1].plot(sorted_durations, cumulative_prob, color='purple', linewidth=2)
    axes[1, 1].set_title(f'{kernel_type} Cumulative Distribution Function')
    axes[1, 1].set_xlabel('Duration')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f"{output_dir}/{kernel_type.lower()}_performance_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def create_performance_plots_with_filename(df_target, kernel_type, output_filename, cluster_labels=None):
    """Create performance analysis charts with custom filename, optionally with cluster information"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{kernel_type} Kernel Performance Deep Analysis', fontsize=16, fontweight='bold')
    
    # 1. Duration distribution histogram with frequency labels (from performance_analysis)
    n, bins, patches = axes[0, 0].hist(df_target['Duration'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title(f'{kernel_type} Duration Distribution')
    axes[0, 0].set_xlabel('Duration')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add frequency labels on top of bars
    for i, (count, bin_center) in enumerate(zip(n, bins[:-1] + (bins[1] - bins[0])/2)):
        if count > 0:  # Only show labels for non-zero bars
            axes[0, 0].text(bin_center, count + max(n) * 0.01, 
                           f'{int(count)}', 
                           ha='center', va='bottom', 
                           fontsize=8, rotation=0)
    
    # 2. Box plot by cluster (from cluster_analysis)
    if cluster_labels is not None:
        # Create cluster renaming map based on mean duration (smallest to largest)
        unique_clusters = sorted(set(cluster_labels))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # Remove noise points
        
        # Calculate mean duration for each cluster and sort
        cluster_means = []
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = df_target[cluster_mask]['Duration']
            cluster_means.append((cluster_id, cluster_data.mean()))
        
        # Sort by mean duration and create renaming map
        cluster_means.sort(key=lambda x: x[1])  # Sort by mean duration
        cluster_rename_map = {}
        for new_id, (old_id, _) in enumerate(cluster_means):
            cluster_rename_map[old_id] = new_id
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(cluster_means)))
        
        # Multi-cluster box plot
        cluster_data_list = []
        cluster_labels_list = []
        for i, (old_cluster_id, _) in enumerate(cluster_means):
            cluster_mask = cluster_labels == old_cluster_id
            cluster_data = df_target[cluster_mask]['Duration']
            new_cluster_id = cluster_rename_map[old_cluster_id]
            cluster_data_list.append(cluster_data.values)
            cluster_labels_list.append(f'Cluster {new_cluster_id}')
        
        box_plot = axes[0, 1].boxplot(cluster_data_list, labels=cluster_labels_list, patch_artist=True)
        
        # Color the boxes to match cluster colors
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    else:
        # Single box plot for non-clustered data
        box_plot = axes[0, 1].boxplot(df_target['Duration'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightcoral')
        box_plot['boxes'][0].set_alpha(0.7)
    
    axes[0, 1].set_title('Duration Distribution by Cluster')
    axes[0, 1].set_ylabel('Duration')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Timeline analysis by cluster (from cluster_analysis)
    if 'Start Timestamp' in df_target.columns:
        timestamps = pd.to_numeric(df_target['Start Timestamp'], errors='coerce')
        if not timestamps.isna().all():
            start_times = timestamps - timestamps.min()
            
            if cluster_labels is not None:
                # Multi-cluster timeline
                for i, (old_cluster_id, _) in enumerate(cluster_means):
                    cluster_mask = cluster_labels == old_cluster_id
                    new_cluster_id = cluster_rename_map[old_cluster_id]
                    axes[1, 0].scatter(start_times[cluster_mask], df_target['Duration'][cluster_mask], 
                                     alpha=0.6, color=colors[i], label=f'Cluster {new_cluster_id}', s=20)
                axes[1, 0].legend()
            else:
                # Single timeline for non-clustered data
                axes[1, 0].scatter(start_times, df_target['Duration'], alpha=0.6, color='green', s=20)
            
            axes[1, 0].set_title('Timeline Analysis by Cluster')
            axes[1, 0].set_xlabel('Relative Time')
            axes[1, 0].set_ylabel('Duration')
            axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cluster statistics table (from cluster_analysis, sorted by mean)
    axes[1, 1].axis('off')
    table_data = []
    headers = ['Cluster', 'Count', 'Mean', 'Std', 'Min', 'Max']
    
    if cluster_labels is not None:
        # Multi-cluster statistics
        for i, (old_cluster_id, mean_duration) in enumerate(cluster_means):
            cluster_mask = cluster_labels == old_cluster_id
            cluster_data = df_target[cluster_mask]['Duration']
            new_cluster_id = cluster_rename_map[old_cluster_id]
            
            table_data.append([
                f'Cluster {new_cluster_id}',
                len(cluster_data),
                f'{cluster_data.mean():.3f}',
                f'{cluster_data.std():.3f}',
                f'{cluster_data.min():.3f}',
                f'{cluster_data.max():.3f}'
            ])
    else:
        # Single cluster statistics for non-clustered data
        table_data.append([
            'All Data',
            len(df_target),
            f'{df_target["Duration"].mean():.3f}',
            f'{df_target["Duration"].std():.3f}',
            f'{df_target["Duration"].min():.3f}',
            f'{df_target["Duration"].max():.3f}'
        ])
    
    table = axes[1, 1].table(cellText=table_data, colLabels=headers, 
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Cluster Statistics Summary')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_filename

def generate_deep_report_with_filename(df, kernel_type, analysis_results, subtypes, output_filename):
    """Generate deep analysis report with custom filename"""
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"# {kernel_type} Kernel Deep Analysis Report\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Basic statistics
        f.write("## 1. Basic Statistics\n\n")
        stats = analysis_results['stats']
        total_duration = analysis_results['total_duration']
        f.write(f"- **Total Call Count**: {int(stats['count'])}\n")
        f.write(f"- **Total Execution Time**: {total_duration:.3f}\n")
        f.write(f"- **Average Execution Time**: {stats['mean']:.3f}\n")
        f.write(f"- **Median Execution Time**: {stats['50%']:.3f}\n")
        f.write(f"- **Minimum Execution Time**: {stats['min']:.3f}\n")
        f.write(f"- **Maximum Execution Time**: {stats['max']:.3f}\n")
        f.write(f"- **Standard Deviation**: {stats['std']:.3f}\n\n")
        
        # Performance tier analysis
        f.write("## 2. Performance Tier Analysis\n\n")
        tiers = analysis_results['performance_tiers']
        total_calls = sum(tiers.values())
        
        f.write("| Performance Tier | Call Count | Percentage | Description |\n")
        f.write("|------------------|------------|------------|-------------|\n")
        for tier, count in tiers.items():
            percentage = (count / total_calls * 100) if total_calls > 0 else 0
            f.write(f"| {tier} | {count} | {percentage:.2f}% | ")
            if tier == 'Fast':
                f.write("Execution time ≤ 50th percentile |\n")
            elif tier == 'Medium':
                f.write("50th < Execution time ≤ 90th percentile |\n")
            elif tier == 'Slow':
                f.write("90th < Execution time ≤ 99th percentile |\n")
            else:
                f.write("Execution time > 99th percentile |\n")
        
        f.write("\n")
        
        # Key performance indicators
        f.write("## 3. Key Performance Indicators\n\n")
        percentiles = analysis_results['percentiles']
        f.write(f"- **P50 (Median)**: {percentiles['p50']:.3f}\n")
        f.write(f"- **P90**: {percentiles['p90']:.3f}\n")
        f.write(f"- **P99**: {percentiles['p99']:.3f}\n")
        f.write(f"- **Coefficient of Variation**: {(stats['std']/stats['mean']*100):.2f}% (std/mean)\n\n")
        
        # Outlier analysis
        f.write("## 4. Outlier Analysis\n\n")
        outliers = analysis_results['outliers']
        f.write(f"- **Outlier Count**: {len(outliers)}\n")
        f.write(f"- **Outlier Percentage**: {(len(outliers)/len(df)*100):.2f}%\n")
        
        if len(outliers) > 0:
            f.write("- **Outlier Statistics**:\n")
            f.write(f"  - Maximum outlier: {outliers['Duration'].max():.3f}\n")
            f.write(f"  - Minimum outlier: {outliers['Duration'].min():.3f}\n")
            f.write(f"  - Average outlier: {outliers['Duration'].mean():.3f}\n")
        
        f.write("\n")
        
        # Subtype analysis
        if subtypes:
            f.write("## 5. Subtype Detailed Analysis\n\n")
            f.write("| Subtype | Call Count | Average Time | Total Time | Percentage |\n")
            f.write("|---------|------------|--------------|------------|------------|\n")
            
            subtype_stats = []
            kernel_total_duration = df['Duration'].sum()
            
            for subtype, rows in subtypes.items():
                subtype_df = pd.DataFrame(rows)
                count = len(subtype_df)
                mean_duration = subtype_df['Duration'].mean()
                total_subtype_duration = subtype_df['Duration'].sum()
                percentage = (total_subtype_duration / kernel_total_duration * 100)
                
                subtype_stats.append({
                    'subtype': subtype,
                    'count': count,
                    'mean': mean_duration,
                    'total': total_subtype_duration,
                    'percentage': percentage
                })
            
            # Sort by total time
            subtype_stats.sort(key=lambda x: x['total'], reverse=True)
            
            for stat in subtype_stats:
                f.write(f"| {stat['subtype']} | {stat['count']} | {stat['mean']:.3f} | {stat['total']:.3f} | {stat['percentage']:.2f}% |\n")
            
            f.write("\n")
        
        # Performance optimization recommendations
        f.write("## 6. Performance Optimization Recommendations\n\n")
        
        if kernel_type == "fMHA":
            f.write("### fMHA Optimization Recommendations:\n")
            f.write("- Consider using the latest version of FlashAttention (v2+)\n")
            f.write("- Optimize sequence length and batch size combinations\n")
            f.write("- Check memory alignment and data layout\n")
            f.write("- Consider using mixed precision training\n")
        elif kernel_type == "GEMM":
            f.write("### GEMM Optimization Recommendations:\n")
            f.write("- Optimize matrix dimensions using powers of 2 or multiples of 8\n")
            f.write("- Consider using Tensor Core optimized GEMM implementations\n")
            f.write("- Check batch operation efficiency\n")
            f.write("- Evaluate performance trade-offs of different precision formats\n")
        elif kernel_type == "NCCL":
            f.write("### NCCL Optimization Recommendations:\n")
            f.write("- Optimize network topology and bandwidth utilization\n")
            f.write("- Consider using NCCL tuning parameters (NCCL_TREE_THRESHOLD, NCCL_IB_DISABLE)\n")
            f.write("- Check for network congestion and latency issues\n")
            f.write("- Consider using NCCL_ASYNC_ERROR_HANDLING for better error recovery\n")
            f.write("- Evaluate if using NCCL_P2P_DISABLE or NCCL_IB_DISABLE improves performance\n")
            f.write("- Consider optimizing data transfer sizes and batch operations\n")
        
        # Performance anomaly checks
        cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0
        if cv > 1.0:
            f.write(f"- ⚠️ **High Variability Warning**: Coefficient of variation {cv:.2f} > 1.0, execution time is unstable\n")
        
        if len(outliers) / len(df) > 0.05:
            f.write(f"- ⚠️ **Outlier Warning**: {(len(outliers)/len(df)*100):.1f}% of calls are outliers\n")
        
        f.write("\n")
        
        # Time distribution characteristics
        f.write("## 7. Time Distribution Characteristics\n\n")
        skewness = df['Duration'].skew()
        kurtosis = df['Duration'].kurtosis()
        
        f.write(f"- **Skewness**: {skewness:.3f} ")
        if skewness > 1:
            f.write("(right-skewed, few high-latency calls exist)")
        elif skewness < -1:
            f.write("(left-skewed, most calls have high latency)")
        else:
            f.write("(approximately normal distribution)")
        f.write("\n")
        
        f.write(f"- **Kurtosis**: {kurtosis:.3f} ")
        if kurtosis > 3:
            f.write("(peaked distribution, high concentration)")
        elif kurtosis < 3:
            f.write("(flat distribution, high dispersion)")
        else:
            f.write("(normal distribution)")
        f.write("\n\n")
    
    return output_filename

def analyze_performance_patterns(df_target, kernel_type):
    """Analyze performance patterns and outliers"""
    duration_stats = df_target['Duration'].describe()
    
    # Calculate outliers
    Q1 = duration_stats['25%']
    Q3 = duration_stats['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_target[(df_target['Duration'] < lower_bound) | 
                        (df_target['Duration'] > upper_bound)]
    
    # Performance tier analysis
    p50 = duration_stats['50%']
    p90 = df_target['Duration'].quantile(0.9)
    p99 = df_target['Duration'].quantile(0.99)
    
    performance_tiers = {
        'Fast': len(df_target[df_target['Duration'] <= p50]),
        'Medium': len(df_target[(df_target['Duration'] > p50) & (df_target['Duration'] <= p90)]),
        'Slow': len(df_target[(df_target['Duration'] > p90) & (df_target['Duration'] <= p99)]),
        'Very Slow': len(df_target[df_target['Duration'] > p99])
    }
    
    # Calculate sum for stats (pandas describe() doesn't include sum)
    total_duration = df_target['Duration'].sum()
    
    return {
        'stats': duration_stats,
        'total_duration': total_duration,
        'outliers': outliers,
        'performance_tiers': performance_tiers,
        'percentiles': {
            'p50': p50,
            'p90': p90,
            'p99': p99
        }
    }

def generate_deep_report(df, kernel_type, analysis_results, subtypes, output_dir):
    """Generate deep analysis report"""
    output_filename = f"{output_dir}/{kernel_type.lower()}_deep_analysis.md"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"# {kernel_type} Kernel Deep Analysis Report\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Basic statistics
        f.write("## 1. Basic Statistics\n\n")
        stats = analysis_results['stats']
        total_duration = analysis_results['total_duration']
        f.write(f"- **Total Call Count**: {int(stats['count'])}\n")
        f.write(f"- **Total Execution Time**: {total_duration:.3f}\n")
        f.write(f"- **Average Execution Time**: {stats['mean']:.3f}\n")
        f.write(f"- **Median Execution Time**: {stats['50%']:.3f}\n")
        f.write(f"- **Minimum Execution Time**: {stats['min']:.3f}\n")
        f.write(f"- **Maximum Execution Time**: {stats['max']:.3f}\n")
        f.write(f"- **Standard Deviation**: {stats['std']:.3f}\n\n")
        
        # Performance tier analysis
        f.write("## 2. Performance Tier Analysis\n\n")
        tiers = analysis_results['performance_tiers']
        total_calls = sum(tiers.values())
        
        f.write("| Performance Tier | Call Count | Percentage | Description |\n")
        f.write("|------------------|------------|------------|-------------|\n")
        for tier, count in tiers.items():
            percentage = (count / total_calls * 100) if total_calls > 0 else 0
            f.write(f"| {tier} | {count} | {percentage:.2f}% | ")
            if tier == 'Fast':
                f.write("Execution time ≤ 50th percentile |\n")
            elif tier == 'Medium':
                f.write("50th < Execution time ≤ 90th percentile |\n")
            elif tier == 'Slow':
                f.write("90th < Execution time ≤ 99th percentile |\n")
            else:
                f.write("Execution time > 99th percentile |\n")
        
        f.write("\n")
        
        # Key performance indicators
        f.write("## 3. Key Performance Indicators\n\n")
        percentiles = analysis_results['percentiles']
        f.write(f"- **P50 (Median)**: {percentiles['p50']:.3f}\n")
        f.write(f"- **P90**: {percentiles['p90']:.3f}\n")
        f.write(f"- **P99**: {percentiles['p99']:.3f}\n")
        f.write(f"- **Coefficient of Variation**: {(stats['std']/stats['mean']*100):.2f}% (std/mean)\n\n")
        
        # Outlier analysis
        f.write("## 4. Outlier Analysis\n\n")
        outliers = analysis_results['outliers']
        f.write(f"- **Outlier Count**: {len(outliers)}\n")
        f.write(f"- **Outlier Percentage**: {(len(outliers)/len(df)*100):.2f}%\n")
        
        if len(outliers) > 0:
            f.write("- **Outlier Statistics**:\n")
            f.write(f"  - Maximum outlier: {outliers['Duration'].max():.3f}\n")
            f.write(f"  - Minimum outlier: {outliers['Duration'].min():.3f}\n")
            f.write(f"  - Average outlier: {outliers['Duration'].mean():.3f}\n")
        
        f.write("\n")
        
        # Subtype analysis
        if subtypes:
            f.write("## 5. Subtype Detailed Analysis\n\n")
            f.write("| Subtype | Call Count | Average Time | Total Time | Percentage |\n")
            f.write("|---------|------------|--------------|------------|------------|\n")
            
            subtype_stats = []
            kernel_total_duration = df['Duration'].sum()
            
            for subtype, rows in subtypes.items():
                subtype_df = pd.DataFrame(rows)
                count = len(subtype_df)
                mean_duration = subtype_df['Duration'].mean()
                total_subtype_duration = subtype_df['Duration'].sum()
                percentage = (total_subtype_duration / kernel_total_duration * 100)
                
                subtype_stats.append({
                    'subtype': subtype,
                    'count': count,
                    'mean': mean_duration,
                    'total': total_subtype_duration,
                    'percentage': percentage
                })
            
            # Sort by total time
            subtype_stats.sort(key=lambda x: x['total'], reverse=True)
            
            for stat in subtype_stats:
                f.write(f"| {stat['subtype']} | {stat['count']} | {stat['mean']:.3f} | {stat['total']:.3f} | {stat['percentage']:.2f}% |\n")
            
            f.write("\n")
        
        # Performance optimization recommendations
        f.write("## 6. Performance Optimization Recommendations\n\n")
        
        if kernel_type == "fMHA":
            f.write("### fMHA Optimization Recommendations:\n")
            f.write("- Consider using the latest version of FlashAttention (v2+)\n")
            f.write("- Optimize sequence length and batch size combinations\n")
            f.write("- Check memory alignment and data layout\n")
            f.write("- Consider using mixed precision training\n")
        elif kernel_type == "GEMM":
            f.write("### GEMM Optimization Recommendations:\n")
            f.write("- Optimize matrix dimensions using powers of 2 or multiples of 8\n")
            f.write("- Consider using Tensor Core optimized GEMM implementations\n")
            f.write("- Check batch operation efficiency\n")
            f.write("- Evaluate performance trade-offs of different precision formats\n")
        elif kernel_type == "NCCL":
            f.write("### NCCL Optimization Recommendations:\n")
            f.write("- Optimize network topology and bandwidth utilization\n")
            f.write("- Consider using NCCL tuning parameters (NCCL_TREE_THRESHOLD, NCCL_IB_DISABLE)\n")
            f.write("- Check for network congestion and latency issues\n")
            f.write("- Consider using NCCL_ASYNC_ERROR_HANDLING for better error recovery\n")
            f.write("- Evaluate if using NCCL_P2P_DISABLE or NCCL_IB_DISABLE improves performance\n")
            f.write("- Consider optimizing data transfer sizes and batch operations\n")
        
        # Performance anomaly checks
        cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0
        if cv > 1.0:
            f.write(f"- ⚠️ **High Variability Warning**: Coefficient of variation {cv:.2f} > 1.0, execution time is unstable\n")
        
        if len(outliers) / len(df) > 0.05:
            f.write(f"- ⚠️ **Outlier Warning**: {(len(outliers)/len(df)*100):.1f}% of calls are outliers\n")
        
        f.write("\n")
        
        # Time distribution characteristics
        f.write("## 7. Time Distribution Characteristics\n\n")
        skewness = df['Duration'].skew()
        kurtosis = df['Duration'].kurtosis()
        
        f.write(f"- **Skewness**: {skewness:.3f} ")
        if skewness > 1:
            f.write("(right-skewed, few high-latency calls exist)")
        elif skewness < -1:
            f.write("(left-skewed, most calls have high latency)")
        else:
            f.write("(approximately normal distribution)")
        f.write("\n")
        
        f.write(f"- **Kurtosis**: {kurtosis:.3f} ")
        if kurtosis > 3:
            f.write("(peaked distribution, high concentration)")
        elif kurtosis < 3:
            f.write("(flat distribution, high dispersion)")
        else:
            f.write("(normal distribution)")
        f.write("\n\n")
    
    return output_filename

def main(args):
    df = pd.read_csv(args.csv)
    df = df[KEEP_SRC_COLUMNS]
    df = df.rename(columns={"Duration Value": "Duration"})
    
    # Add kernel type
    df['Type'] = df.apply(determine_kernel_type, axis=1)
    
    # Convert Duration to numeric
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    df_clean = df.dropna(subset=['Duration'])
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    plot_subdir = f"{args.output_dir}/plots"
    report_subdir = f"{args.output_dir}/reports"
    os.makedirs(plot_subdir, exist_ok=True)
    os.makedirs(report_subdir, exist_ok=True)
    
    # Analyze target kernel types
    target_types = ['fMHA', 'GEMM', 'NCCL']

    if "sp1" in args.csv:
        min_cluster_size = 2
    else:
        min_cluster_size = MIN_CLUSTER_SIZE
    
    for kernel_type in target_types:
        df_target = df_clean[df_clean['Type'] == kernel_type]
        
        if df_target.empty:
            print(f"Warning: No {kernel_type} kernel data found")
            continue
        
        print(f"\nAnalyzing {kernel_type} kernel type...")
        print(f"Found {len(df_target)} {kernel_type} kernel calls")
        
        # Subtype analysis
        if kernel_type == 'fMHA':
            subtypes = analyze_fmha_subtypes(df_target)
        elif kernel_type == 'GEMM':
            subtypes = analyze_gemm_subtypes(df_target)
        elif kernel_type == 'NCCL':
            subtypes = analyze_nccl_subtypes(df_target)
        else:
            subtypes = {}
        
        # Initialize clustering variables and prepare file naming
        cluster_labels = None
        clustering_performed = False
        base_filename = get_output_filename_base(args.csv)
        
        # Special clustering analysis for fMHA, GEMM, and NCCL
        if kernel_type in ['fMHA', 'GEMM', 'NCCL'] and SKLEARN_AVAILABLE:
            print(f"Performing clustering analysis for {kernel_type} kernels...")
            cluster_labels, n_clusters, silhouette_score_val = perform_clustering_analysis(df_target, kernel_type, min_cluster_size)

            if cluster_labels is not None:
                print(f"Initial clustering found {n_clusters} clusters with silhouette score: {silhouette_score_val:.3f}")
                
                # Merge similar clusters with different thresholds for different kernel types
                if kernel_type == 'fMHA':
                    merge_threshold = 0.1  # 10% for fMHA
                elif kernel_type == 'GEMM':
                    merge_threshold = 0.1  # 10% for GEMM
                elif kernel_type == 'NCCL':
                    merge_threshold = 0.15  # 15% for NCCL (communication can be more variable)
                else:
                    merge_threshold = 0.1  # Default
                
                print(f"Using merge threshold of {merge_threshold*100:.1f}% for {kernel_type}")
                
                merged_labels, final_n_clusters = merge_similar_clusters(df_target, cluster_labels, threshold_percent=merge_threshold)

                if final_n_clusters != n_clusters:
                    print(f"After merging similar clusters: {final_n_clusters} final clusters")
                    cluster_labels = merged_labels
                    n_clusters = final_n_clusters
                
                # Validate clustering quality
                # Always use clustering regardless of validation
                print(f"Clustering forced: proceeding with {n_clusters} clusters")
                # Show cluster statistics for information
                validate_clustering_quality(df_target, cluster_labels, kernel_type)
            elif kernel_type == 'GEMM':
                # For GEMM, if initial clustering failed, force a basic 2-cluster analysis
                print("Initial clustering failed for GEMM. Forcing basic 2-cluster analysis...")
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(df_target[['Duration']])
                    n_clusters = 2
                    silhouette_score_val = 0.0  # Placeholder since we're forcing it
                    print(f"Forced GEMM clustering: 2 clusters created")
                except:
                    print("Failed to force GEMM clustering. This should not happen.")
                    cluster_labels = None
            elif kernel_type == 'NCCL':
                # For NCCL, if initial clustering failed, force a basic 2-cluster analysis
                print("Initial clustering failed for NCCL. Forcing basic 2-cluster analysis...")
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(df_target[['Duration']])
                    n_clusters = 2
                    silhouette_score_val = 0.0  # Placeholder since we're forcing it
                    print(f"Forced NCCL clustering: 2 clusters created")
                except:
                    print("Failed to force NCCL clustering. This should not happen.")
                    cluster_labels = None
                
            # For GEMM and NCCL, ensure we always have clustering; for others, check validation
            if cluster_labels is not None or kernel_type in ['GEMM', 'NCCL']:
                if kernel_type == 'GEMM' and cluster_labels is None:
                    print("🔧 GEMM emergency fallback: creating simple binary clustering...")
                    # Create a simple binary clustering based on median
                    median_duration = df_target['Duration'].median()
                    cluster_labels = (df_target['Duration'] > median_duration).astype(int)
                    n_clusters = 2
                    silhouette_score_val = 0.0
                    print(f"✅ Emergency GEMM clustering created: 2 clusters (above/below median)")
                elif kernel_type == 'NCCL' and cluster_labels is None:
                    print("🔧 NCCL emergency fallback: creating simple binary clustering...")
                    # Create a simple binary clustering based on median
                    median_duration = df_target['Duration'].median()
                    cluster_labels = (df_target['Duration'] > median_duration).astype(int)
                    n_clusters = 2
                    silhouette_score_val = 0.0
                    print(f"✅ Emergency NCCL clustering created: 2 clusters (above/below median)")
                
                if cluster_labels is not None:  # Final check
                    # Generate unified performance analysis with cluster information
                    plot_file = f"{plot_subdir}/{base_filename}.{kernel_type.lower()}_performance_analysis.png"
                    create_performance_plots_with_filename(df_target, kernel_type, plot_file, cluster_labels)
                    print(f"Unified performance analysis chart with cluster information saved: {plot_file}")
                    
                    # Generate unified cluster analysis report
                    unified_report_file = f"{report_subdir}/{base_filename}.{kernel_type.lower()}_deep_analysis.md"
                    generate_unified_cluster_report(df_target, cluster_labels, n_clusters, silhouette_score_val, subtypes, kernel_type, unified_report_file)
                    print(f"Unified cluster analysis report saved: {unified_report_file}")
                    
                    clustering_performed = True
                else:
                    print(f"Clustering validation failed for {kernel_type}, falling back to standard analysis")
            
            else:
                print("No meaningful clusters found, proceeding with standard analysis")
        
        # Standard performance pattern analysis (for all kernel types)
        analysis_results = analyze_performance_patterns(df_target, kernel_type)
        
        # Only generate standard reports if clustering was not performed (or not applicable)
        if not clustering_performed:
            plot_file = f"{plot_subdir}/{base_filename}.{kernel_type.lower()}_performance_analysis.png"
            create_performance_plots_with_filename(df_target, kernel_type, plot_file)
            print(f"Performance chart saved: {plot_file}")
            
            # Generate deep analysis report
            report_file = f"{report_subdir}/{base_filename}.{kernel_type.lower()}_deep_analysis.md"
            generate_deep_report_with_filename(df_target, kernel_type, analysis_results, subtypes, report_file)
            print(f"Deep analysis report saved: {report_file}")
    
    print(f"\nDeep analysis completed! All results saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep performance analysis for fMHA, GEMM, and NCCL kernels")
    parser.add_argument("-c", "--csv", type=str, required=True, help="CSV file to analyze")
    parser.add_argument("--output-dir", type=str, default="./deep_analysis_results", 
                       help="Directory to save analysis results")
    args = parser.parse_args()
    main(args)
