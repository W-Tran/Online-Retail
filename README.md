# Customer Segmentation

<p align="center">
  <img src="data/figures/monetary_cluster_rf.png" width="400">
  <img src="data/figures/monetary_cluster_pca.png" width="400">
</p>
<p align="center"><img src="data/figures/typical_customer.png" width=800></p>
<p align="center"><img src="data/figures/pareto.png" width=400></p>

An analysis of 2 years of transaction history data from a real UK based online retailer. Both years of the dataset can be found on the UCI Machine Learning Repository ([first year](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II), [second year](https://archive.ics.uci.edu/ml/datasets/Online+Retail)).

The analysis shows that highly profitable customer value groups can be identified by segmenting customers based on their first year purchasing behaviour. The highest value customer groups ended up making the largest number of high value purchases in their second year of purchasing. Attempting to segment customers into value groups based on their RFM statistics resulted in notable second year outliers which were customers identified to be low value but ended up purchasing a large amount in their second year. Using a GMM allows for a soft segmentation where customers are assigned cluster label probabilities which is useful for distinguishing between customers who are most likely to be high value. 

**Notes/Future Work**

- Customers were segmented based on their first year transactions whilst the segmentation was evaluated based on their second year transactions.
- Other features could be used to segment the customers depending on the marketing strategy and what customer behaviour is trying to be understood. Due to limitations of the dataset, purchasing behaviour was the only features that could be produced.
- Other clustering algorithms could be experimented with to produce different segmentation geometry that might cluster the customer base into more useful groups. 

## Installation

Simply clone the repo and install all dependencies listed in the requirements.txt file to an environment of your choice.

## Usage

All results and plots can be reproduced using the clustering_daily.ipynb notebook.
