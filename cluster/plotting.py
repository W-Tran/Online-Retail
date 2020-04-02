import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def cluster_elbow_method(cluster_features):
    sse = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, n_init=200, max_iter=1000).fit(cluster_features)
        sse.append(kmeans.inertia_)

    plt.plot(list(range(1, 10)), sse)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Intertia")


def cluster_rf_plot(cluster_features, features):
    if "recency" not in cluster_features:
        cluster_features['recency'] = features['recency'].values
    if "frequency" not in cluster_features:
        cluster_features['frequency'] = features['frequency'].values
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    sns.scatterplot(
        'recency',
        'frequency',
        hue='Cluster',
        size='SecondYearRLV',
        palette='rainbow',
        sizes=(0, 1000),
        alpha=0.8,
        data=cluster_features,
        ax=ax
    )
    ax.grid(alpha=0.2)
    ax.set_facecolor("snow")


def cluster_pca_plot(cluster_features, cluster_centers, scaled=True):
    cluster_centers = cluster_centers.copy()
    cluster_feature_values = cluster_features.iloc[:, :-2].copy()
    if not scaled:
        scaler = StandardScaler()
        cluster_feature_values = scaler.fit_transform(cluster_feature_values)
        cluster_centers = scaler.transform(cluster_centers)
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(cluster_feature_values)
    pca_components_df = pd.DataFrame(np.column_stack((pca_components, cluster_features['Cluster'].values)),
                                     columns=['PC1', 'PC2', 'Cluster'])
    pca_components_df['Cluster'] = pca_components_df['Cluster'].astype(int)
    pca_components_df['SecondYearRLV'] = cluster_features['SecondYearRLV']

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]}")

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    sns.scatterplot(
        'PC1',
        'PC2',
        hue='Cluster',
        size='SecondYearRLV',
        palette='rainbow',
        sizes=(0, 1000),
        data=pca_components_df,
        ax=ax
    )
    ax.grid(alpha=0.2)
    ax.set_facecolor("snow")
    plt.scatter(
        pca.transform(cluster_centers)[:, 0],
        pca.transform(cluster_centers)[:, 1],
        marker='x',
        color='k',
        s=100,
    )

    return pca


def pareto_plot(df, x=None, y=None, title=None, show_pct_y=False, pct_format='{0:.0%}'):
    xlabel = x
    ylabel = y
    tmp = df.sort_values(y, ascending=False)
    x = tmp[x].values
    y = tmp[y].values
    weights = y / y.sum()
    cumsum = weights.cumsum()

    fig, ax1 = plt.subplots()
    ax1.bar(x, y)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax2 = ax1.twinx()
    ax2.plot(x, cumsum, '-ro', alpha=0.5)
    ax2.set_ylabel('', color='r')
    ax2.tick_params('y', colors='r')

    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

    # hide y-labels on right side
    if not show_pct_y:
        ax2.set_yticks([])

    formatted_weights = [pct_format.format(x) for x in cumsum]
    for i, txt in enumerate(formatted_weights):
        ax2.annotate(txt, (x[i], cumsum[i]), fontweight='heavy')

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()
