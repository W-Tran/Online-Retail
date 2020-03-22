import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.mixture import GaussianMixture
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.metrics import silhouette_score


def get_money_aggregation_features(cohort_invoices):
    cohort_invoices = cohort_invoices.copy()
    daily_customer_revenues = cohort_invoices.groupby(["CustomerID", pd.Grouper(key="InvoiceDate", freq="D")])[
        'Revenue'].sum().reset_index()
    aggregation_features = daily_customer_revenues.groupby('CustomerID')['Revenue'].agg([
        'mean',
        'std',
        'min',
        'max',
        'sum',
        'size',
    ]).reset_index()

    # Rename features
    aggregation_features.columns = [
        'CustomerID',
        'MeanPurchaseValue',
        'StDevPurchaseValue',
        'MinPurchaseValue',
        'MaxPurchaseValue',
        'SumPurchaseValue',
        'SizePurchaseValue',
    ]
    # Customers who purchased more than once in the first year
    aggregation_features.dropna(inplace=True)

    return aggregation_features


def transform_features(features, feats_to_transform, transform="log"):
    features = features.copy()
    transformed_feat_names = []
    if transform == "log":
        for feat in feats_to_transform:
            features[feat] = np.log1p(features[feat])
            features.rename(columns={feat: f'log(1+{feat})'}, inplace=True)
            transformed_feat_names.append(f'log(1+{feat})')
    elif transform == "yjt":
        yjt = PowerTransformer(method='yeo-johnson', standardize=False)
        for feat in feats_to_transform:
            features[feat] = yjt.fit_transform(features[feat].to_frame())
            features.rename(columns={feat: f'yjt_{feat}'}, inplace=True)
            transformed_feat_names.append(f'yjt_{feat}')
    return features, transformed_feat_names


def get_rfm_features(features, cohort_invoices):
    cohort_invoices = cohort_invoices.copy()
    features = features.copy()

    rfm_features = summary_data_from_transaction_data(
        transactions=cohort_invoices,
        customer_id_col='CustomerID',
        datetime_col='InvoiceDate',
        monetary_value_col='Revenue',
        freq='D'
    ).reset_index()
    features = features.merge(rfm_features, how='left', on='CustomerID')
    features['T_Minus_Recency'] = rfm_features['T'] - rfm_features['recency']

    return features


def get_total_first_year_revenue(features, cohort_invoices):
    cohort_invoices = cohort_invoices.copy()
    features = features.copy()

    features = features.merge(
        cohort_invoices.groupby('CustomerID')['Revenue'].sum().reset_index(),
        how='left',
        on='CustomerID'
    )
    features.rename(columns={'Revenue': 'TotalFirstYearRevenue'}, inplace=True)
    features['log(1+TotalFirstYearRevenue)'] = np.log1p(features['TotalFirstYearRevenue'])

    return features


def get_second_year_rlv(features, cohort_invoices):
    cohort_invoices = cohort_invoices.copy()
    features = features.copy()

    second_year_cohort_revenues = cohort_invoices.groupby('CustomerID')['Revenue'].sum().reset_index()
    second_year_cohort_revenues.columns = ['CustomerID', 'SecondYearRLV']
    features = features.merge(second_year_cohort_revenues, how='left', on='CustomerID')
    features.fillna(value=0, inplace=True)

    return features


def perform_km_clustering(features, cluster_feats, num_clusters, scaler='mm'):
    features = features.copy()
    if scaler == 'mm':
        scaler = MinMaxScaler()
    elif scaler == 'normal':
        scaler = StandardScaler()
    scaled_monetary_feats = scaler.fit_transform(features[cluster_feats])
    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000, n_init=200, n_jobs=4)
    kmeans.fit(scaled_monetary_feats)
    features['Cluster'] = kmeans.labels_
    features['Cluster'] = features['Cluster'].astype(int)
    print(f"Inertia for {num_clusters} clusters: {kmeans.inertia_}")

    return features, scaler, kmeans.cluster_centers_


def perform_gmm_clustering(features, cluster_feats, num_clusters, scaler='mm'):
    features = features.copy()
    if scaler == 'mm':
        scaler = MinMaxScaler()
    elif scaler == 'normal':
        scaler = StandardScaler()
    scaled_monetary_feats = scaler.fit_transform(features[cluster_feats])
    gmm = GaussianMixture(n_components=num_clusters, covariance_type="full", n_init=25, max_iter=200)
    features['Cluster'] = gmm.fit_predict(scaled_monetary_feats)
    features['Cluster'] = features['Cluster'].astype(int)

    return features, scaler, gmm.means_


def reorder_clusters(features, cluster_centers, num_clusters):
    features = features.copy()

    rlv_centres = []
    for cluster_num in range(num_clusters):
        rlv_centres.append(features[features['Cluster'] == cluster_num].SecondYearRLV.mean())
    cluster_idx_rlv_pairs = list(zip([cluster_num for cluster_num in range(num_clusters)], rlv_centres))

    # Order list of (Cluster, RLV) tuple pairs from smallest to largest RLV
    sorted_idx_rlv_pairs = sorted(cluster_idx_rlv_pairs, key=lambda x: x[1])
    cluster_reorder_mapping = {enum[1][0]: enum[0] for enum in enumerate(sorted_idx_rlv_pairs)}
    features['Cluster'].replace(cluster_reorder_mapping, inplace=True)

    sorted_cluster_centers = np.zeros_like(cluster_centers)
    for old_cluster_idx, new_cluster_idx in cluster_reorder_mapping.items():
        sorted_cluster_centers[new_cluster_idx] = cluster_centers[old_cluster_idx]

    return features, sorted_cluster_centers


def evaluate_clusters(features, num_clusters, cluster_feats, scaler):
    labeled_rlv_centres = []
    rlv_centres = []
    for cluster_num in range(num_clusters):
        labeled_rlv_centres.append((cluster_num,
                                    np.around(features[features['Cluster'] == cluster_num].SecondYearRLV.median()), 2))
        rlv_centres.append(features[features['Cluster'] == cluster_num].SecondYearRLV.median())
    diff = [np.around(abs(j - i), 2) for i, j in zip(rlv_centres, rlv_centres[1:])]

    print(f"Cluster labels - 0 = Low value customer, {num_clusters} = High value customer")
    print(f"Silhouette Coefficient:"
          f" {silhouette_score(scaler.transform(features[cluster_feats]), features['Cluster'], metric='euclidean')}")
    print(f"Median 2nd year RLV: {labeled_rlv_centres}")
    print(f"Differences: {diff}")
    print("Value Counts:")
    print(features.Cluster.value_counts().sort_index())
