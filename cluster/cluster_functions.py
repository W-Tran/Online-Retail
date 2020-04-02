import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.mixture import GaussianMixture
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.metrics import silhouette_score


def get_money_aggregation_features(cohort_invoices):
    """
    Get aggregations of daily purchase value for each customer.
    The "Purchase Value" for a given day is the sum of all transactions made on the same day.
    """
    cohort_invoices = cohort_invoices.copy()
    daily_customer_revenues = cohort_invoices.groupby(["CustomerID", pd.Grouper(key="InvoiceDate", freq="D")])[
        'Revenue'].sum().reset_index()
    aggregation_features = daily_customer_revenues.groupby('CustomerID')['Revenue'].agg([
        'median',  # 'mean',
        'std',
        'min',
        'max',
        'sum',
        'size',
    ]).reset_index()

    # Rename features
    aggregation_features.columns = [
        'CustomerID',
        'MedianPurchaseValue',  # 'MeanPurchaseValue',
        'StDevPurchaseValue',
        'MinPurchaseValue',
        'MaxPurchaseValue',
        'SumPurchaseValue',
        'SizePurchaseValue',
    ]
    aggregation_features.dropna(inplace=True)

    return aggregation_features


def get_time_aggregation_features(features, cohort_invoices):
    """
    Get aggregations of time between purchase days.
    """
    cohort_invoices = cohort_invoices.copy()
    cohort_invoices['InvoiceDay'] = cohort_invoices['InvoiceDate'].dt.date
    cohort_single_daily_invoices = cohort_invoices.drop_duplicates(subset=['CustomerID', 'InvoiceDay'], keep='first')
    cohort_single_daily_invoices = cohort_single_daily_invoices.sort_values(by=['CustomerID', 'InvoiceDate'])

    cohort_single_daily_invoices['PrevInvoiceDate'] = cohort_single_daily_invoices.groupby('CustomerID')[
        'InvoiceDate'].shift(1)
    cohort_single_daily_invoices['TimeBetweenInvoices'] = (
            cohort_single_daily_invoices['InvoiceDate'] - cohort_single_daily_invoices['PrevInvoiceDate']).dt.days
    aggregation_features = cohort_single_daily_invoices.groupby('CustomerID')['TimeBetweenInvoices'].agg([
        'median',  # 'mean',
        'std',
        'min',
        'max'
    ]).reset_index()

    # Rename features
    aggregation_features.columns = [
        'CustomerID',
        'MedianTimeBetweenPurchase',  # 'MeanTimeBetweenPurchase',
        'StDevTimeBetweenPurchase',
        'MinTimeBetweenPurchase',
        'MaxTimeBetweenPurchase'
    ]
    features = features.merge(aggregation_features, how='left', on='CustomerID')
    features = features.dropna()  # Customers who purchased at least 3 times during calibration period

    return features


def transform_features(features, features_to_transform, transform="log"):
    features = features.copy()
    transformed_feature_names = []
    if transform == "log":
        for feat in features_to_transform:
            features[feat] = np.log1p(features[feat])
            features.rename(columns={feat: f'log(1+{feat})'}, inplace=True)
            transformed_feature_names.append(f'log(1+{feat})')
    elif transform == "yjt":
        yjt = PowerTransformer(method='yeo-johnson', standardize=False)
        transformed_features = pd.DataFrame(yjt.fit_transform(features[features_to_transform]),
                                            columns=features_to_transform)
        features.drop(columns=features_to_transform, inplace=True)
        features = pd.concat([transformed_features, features], axis=1)
        for feat in features_to_transform:
            features.rename(columns={feat: f'yjt_{feat}'}, inplace=True)
            transformed_feature_names.append(f'yjt_{feat}')
    return features, transformed_feature_names, yjt if transform == "yjt" else None


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


def get_second_year_rlv(features, cohort_invoices):
    """
    The second year Residual Lifetime Value of each customer
    """
    cohort_invoices = cohort_invoices.copy()
    features = features.copy()

    second_year_cohort_revenues = cohort_invoices.groupby('CustomerID')['Revenue'].sum().reset_index()
    second_year_cohort_revenues.columns = ['CustomerID', 'SecondYearRLV']
    features = features.merge(second_year_cohort_revenues, how='left', on='CustomerID')
    features.fillna(value=0, inplace=True)

    return features


def perform_km_clustering(cluster_features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000, n_init=200, n_jobs=4)
    kmeans.fit(cluster_features)
    cluster_features['Cluster'] = kmeans.labels_
    cluster_features['Cluster'] = cluster_features['Cluster'].astype(int)
    print(f"Inertia for {num_clusters} clusters: {kmeans.inertia_}")

    return cluster_features, kmeans.cluster_centers_


def perform_gmm_clustering(cluster_features, num_clusters):
    gmm = GaussianMixture(
        n_components=num_clusters,
        covariance_type="full",
        n_init=25,
        max_iter=200
    )
    cluster_features['Cluster'] = gmm.fit_predict(cluster_features)
    cluster_features['Cluster'] = cluster_features['Cluster'].astype(int)

    return cluster_features, gmm.means_, gmm


def reorder_clusters(cluster_features, cluster_centers, features, num_clusters):
    """
    Order cluster and cluster center indices in ascending order of mean 2nd year RLV:
    0 = Cluster with lowest second year RLV
    num_clusters-1 = Cluster with highest second year RLV

    This is done to make it easier to identify profitable customers segments from plots.
    If rlv is unknown, observe instead the mean cluster statistics to identify which
    segments are the most profitable.
    """
    cluster_features = cluster_features.copy()
    cluster_features['SecondYearRLV'] = features['SecondYearRLV'].values
    cluster_rlv = []
    for cluster_num in range(num_clusters):
        cluster_rlv.append(cluster_features[cluster_features['Cluster'] == cluster_num].SecondYearRLV.mean())
    cluster_idx_rlv_pairs = list(zip([cluster_num for cluster_num in range(num_clusters)], cluster_rlv))

    # Order list of (Cluster, RLV) tuple pairs from smallest to largest RLV
    sorted_idx_rlv_pairs = sorted(cluster_idx_rlv_pairs, key=lambda x: x[1])
    cluster_reorder_mapping = {enum[1][0]: enum[0] for enum in enumerate(sorted_idx_rlv_pairs)}
    cluster_features['Cluster'].replace(cluster_reorder_mapping, inplace=True)

    sorted_cluster_centers = np.zeros_like(cluster_centers)
    for old_cluster_idx, new_cluster_idx in cluster_reorder_mapping.items():
        sorted_cluster_centers[new_cluster_idx] = cluster_centers[old_cluster_idx]

    return cluster_features, sorted_cluster_centers


def evaluate_clusters(cluster_features, num_clusters):
    """
    Print evaluation statistics for the clusters
    """
    cluster_rlv = []
    for cluster_num in range(num_clusters):
        cluster_rlv.append(cluster_features[cluster_features['Cluster'] == cluster_num].SecondYearRLV.median())
    diff = [np.around(abs(j - i), 2) for i, j in zip(cluster_rlv, cluster_rlv[1:])]

    print(f"Cluster labels - 0 = Low value customer, {num_clusters - 1} = High value customer")
    print(f"Silhouette Coefficient:"
          f"{silhouette_score(cluster_features.iloc[:, :-1], cluster_features['Cluster'], metric='euclidean')}")
    print(f"Median 2nd year RLV: {np.around(cluster_rlv, 3)}")
    print(f"Differences: {diff}")
    print("Value Counts:")
    print(cluster_features['Cluster'].value_counts().sort_index())


def select_best_num_components(features, cluster_features):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cv_type,
                n_init=25,
                max_iter=200
            )
            gmm.fit(features[cluster_features])
            bic.append(gmm.bic(features[cluster_features]))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return gmm.n_components
