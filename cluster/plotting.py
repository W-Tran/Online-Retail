import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def cluster_elbow_method(features, cluster_feats, scaler='mm'):
    sse = []
    if scaler == 'mm':
        scaler = MinMaxScaler()
    elif scaler == 'normal':
        scaler = StandardScaler()

    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, n_init=200, max_iter=1000).fit(scaler.fit_transform(features[cluster_feats]))
        sse.append(kmeans.inertia_)

    plt.plot(list(range(1, 10)), sse)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Intertia")


def cluster_rf_plot(features):
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
        data=features,
        ax=ax
    )
    ax.grid(alpha=0.2)
    ax.set_facecolor("snow")


def cluster_pca_plot(features, cluster_feats, scaler, sorted_cluster_centers):
    pca = PCA(n_components=2)
    scaled_cluster_feats = scaler.transform(features[cluster_feats])
    pca_cluster_feats = pca.fit_transform(scaled_cluster_feats)

    pca_cluster_feats_df = pd.DataFrame(np.column_stack((pca_cluster_feats, features['Cluster'].values)),
                                        columns=['PC1', 'PC2', 'Cluster'])
    pca_cluster_feats_df['Cluster'] = pca_cluster_feats_df['Cluster'].astype(int)
    pca_cluster_feats_df['SecondYearRLV'] = features['SecondYearRLV']
    pca_cluster_feats_df['CustomerID'] = features['CustomerID']

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
        data=pca_cluster_feats_df,
        ax=ax
    )
    ax.grid(alpha=0.2)
    ax.set_facecolor("snow")
    plt.scatter(
        pca.transform(sorted_cluster_centers)[:, 0],
        pca.transform(sorted_cluster_centers)[:, 1],
        marker='x',
        color='k',
        s=100,
    )

    return pca


def cluster_tsne_plot(features, cluster_feats, scaler, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    scaled_cluster_feats = scaler.transform(features[cluster_feats])
    tsne_cluster_feats = tsne.fit_transform(scaled_cluster_feats)

    tsne_cluster_feats_df = pd.DataFrame(np.column_stack((tsne_cluster_feats, features['Cluster'].values)),
                                         columns=['Component 1', 'Component 2', 'Cluster'])
    tsne_cluster_feats_df['Cluster'] = tsne_cluster_feats_df['Cluster'].astype(int)
    tsne_cluster_feats_df['SecondYearRLV'] = features['SecondYearRLV']
    tsne_cluster_feats_df['CustomerID'] = features['CustomerID']

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    sns.scatterplot(
        'Component 1',
        'Component 2',
        hue='Cluster',
        size='SecondYearRLV',
        palette='rainbow',
        sizes=(0, 1000),
        data=tsne_cluster_feats_df,
        ax=ax
    )
    ax.grid(alpha=0.2)
    ax.set_facecolor("snow")

    return tsne


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