# Customer Segmentation

<p align="center">
  <img src="data/figures/monetary_cluster_rf.png" width="400">
  <img src="data/figures/monetary_cluster_pca.png" width="400">
</p>
<p align="center"><img src="data/figures/typical_customer.png" width=800></p>
<p align="center"><img src="data/figures/pareto.png" width=400></p>

An analysis of 2 years of transaction history data from a real UK based online retail. Both years of the dataset can be found on the UCI Machine Learning Repository ([first year](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II), [second year](https://archive.ics.uci.edu/ml/datasets/Online+Retail)).

The notebook shows that clustering customers based on their RFM isn't able to group together high second year spenders as well as simply clustering on monetary daily aggregate statistics such as Mean/Min/Max/Sum/StDev revenue per transaction day (multiple purchases on the same day were grouped together into 1 transaction).

**Notes/Future Work**

- Customers were clustered based on their first year transactions whilst the segmentation was evaluated based on their second year transactions.
- Other features could also be added to the clustering feature set depending on what retailers believe to be important when segmenting their customers. For example a similar segmentation to the RFM clustering could be achieved by adding recency/frequency statistics to the aggregate features. This will recreate the banding seen in the RFM clustering RF plot.
- Other clustering algorithms could be experimented with to produce different segmentation geometry that might cluster the customer base into more useful groups.

## Installation

Simply clone the repo and install all the dependencies listed in the requirements.txt file. All results and plots can be reproduced using the clustering_daily.ipynb notebook.
