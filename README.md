# Introduction

Data-driven performance evaluation of players is becoming more widely used in the soccer industry[2, 3, 4, 15]. For a team, the ability to compare players is critical to success as it provides decision support to many areas, e.g. which athletes to include on match day rosters, which to start and who to substitute, who to target for roster building during the transfer windows for professionals or who to recruit for collegiate teams[1]. As such, there is great interest in the creation of a data-driven performance rating metric that quantifies the quality of a player's performance based on related spatio-temporal events during a match. The rating can then be used for a direct comparison of players.

One such rating, PlayeRank, proposed by Pappalardo et al. [15], was groundbreaking in three areas: it was the first multi-dimensional rating approach; the first to consider the specificity of each player's role on the field; and the first to evaluate the goodness of ranking in a quantitative manner similar to recommender systems in Information Retrieval. The authors offer several areas for potential improvement, one specifically mentioned is development of improved methods for identifying a player's role within a match or a specific portion of it. Due to the modular nature of PlayeRank, other methods of role detection can be seamlessly integrated allowing users to personalize role detection based on their individual requirements [15].

In this paper we will follow in the footsteps of Pappalardo et al. [15] with regard to the player role detection. We will attempt to verify and potentially improve upon their results through the use of unsupervised machine learning methods to identify a set of player roles utilizing match events in a soccer-logs dataset.

# Related Works

The advent of extensive data sets depicting soccer performance has greatly contributed to the recent progress in soccer analytics. Soccer-logs, comprehensive records of all events taking place during a match, have emerged as a prevalent data format, enabling analysis of various aspects of soccer at both team and individual levels[14, 15]. One aspect of PlayeRank that is of particular interest is role detection[15].

### Performance Metrics and Evaluation

Prior to PlayeRank, there were various approaches to evaluating player performance. One of the pioneering efforts was the flow centrality (FC) metric, a network approach that quantifies individual player contributions and overall team performance. It provided a comprehensive evaluation of player performance and its impact on team outcomes[4]. Another interesting metric was the Pass Shot Value (PSV) metric, which focuses on the value of completed passes and utilizes a supervised machine learning model to analyze pass locations and their relationship to generating shot opportunities[2]. However, both of these metrics struggled to be adequate when applied outside of attacking players, i.e. defending players. PlayeRank, follows those earlier attempts, and sought to overcome their shortcomings. It is a multi-dimensional, data-driven framework and provides role-aware evaluation of player performance[15].

### Player Role Recognition through Spatio-Temporal Events

One of the problems that plagued early attempts at data-driven player evaluation was lack of role awareness. Early attempts did not distinguish between where players were located on the pitch and what tasks they were assigned within the team's tactical structure. These approach shortcomings were noted by their developers[2, 4]. PlayeRank was the first metric that proposed a method for producing a role-aware player evaluation[15]. Outside of the sport of soccer, other methods existed for role detection via spatio-temporal events[6, 18]. However, these techniques were found inadequate when applied to soccer due to its lower event density [15].

# Methodology

This project aims to reproduce the role detection results found by Pappalardo et al. [15]. In order to meaningfully compare and verify those findings, we adopt the same overall average silhouette width $\bar{s}_{k}$, or silhouette score, to validate our methods. This is the mean of $s_{i}$ for all $i$ in the data set, with $s_i$ defined as: $$s_{i} =\frac{b_i - a_i}{\textsf{max}(a_i, b_i)}$$ where, $a_i$ is the mean intra-cluster dissimilarity, and $b_i$ is the mean nearest-cluster dissimilarity [17].

We make use of a public data set of soccer-logs from Wyscout consisting of 3,251,294 events from 1,941 matches and 4,299 players. This data set includes the 2017/18 seasons of the first division competitions in England, France, Germany, Italy, and Spain, as well as the 2018 Fédération Internationale de Football Association (FIFA) World Cup and the 2016 Union of European Football Associations (UEFA) European Football Championship [14]. Each event record consists of twelve items describing seven characteristics, of which the following four were of primary interest in our study:

```{=tex}
\begin{enumerate}
  \item eventId, eventName, subEventId, subEventName, tags: type of event and additional information
  \item matchId: identifier for the match
  \item playerId: identifier for player who generated the event
  \item positions: $(x, y)$ coordinate pair origin and destination positions of the event $(x,y) \in [0, 100]$ ($x$ and $y$ indicate event's nearness to opponent's goal and nearness to right side of the pitch, respectively, in percentage)
\end{enumerate}
```
We omit goalkeeping events and goalkeepers from the analysis as the role has different rules from other players. After removing events originating with goalkeepers, there were 2,873,772 events remaining which resulted in a total of 49,495 player-match groups. The distribution of these observations is shown in @fig-events. The mean number of events per player per match was $\mu = 73$ with a standard deviation of $\sigma = 26.5$, a minimum of $1$ and a maximum of $225$.

:::{#fig-events}

![](/figs/p_events.png)

Distribution of Events by Player-Match Group
:::

In order to achieve our goal, we implement the same method introduced by Pappalardo et al. [15]. This method calculates a *center of performance* for each player in each match by average position $(\bar{x}, \bar{y})$ of events generated by the player $u$ in a given match $m$, such that a *center of performance* for a given player and match is denoted $c_u^m = (\bar{x}, \bar{y})$. A soft-clustering is then applied by k-means over the *centers of performance* of all players in all matches. The soft-clustering allows for "hybrid" roles, where the *center of performance* is assigned to two or more clusters based on a tolerance threshold, $\delta_{s}$.

This is accomplished by calculating the silhouette width in reference to all clusters such that for any *center of performance*, $i$, in cluster $C_i$ the silhouette for each cluster $C_k$, where $C_k \ne C_i$, is calculated: $$ s_{iC_k} = \frac{d_i - a_i}{\textsf{max}(a_i, d_i)} $$ where $d_i$ is the mean nearest-cluster dissimilarity to cluster $C_k$. $i$ is assigned to all clusters $C_k$ where $s_{iC_k} \in (0, \delta_{s})$. The silhouette widths are calculated again using the standard method. For hybrid $i$, a average silhouette width is calculated as $\sum_{j=1}^n{s_{ij}} / n$, where $n$ is the total number of clusters $i$ is a member. Papplardo et al. varied $k = 1,\ldots,20$ with $\delta_{s} = 0.1$, or $5%$ of centers classified as "hybrid". They found that $k = 8$ performed best with a silhouette score of $0.43$. We will omit $k = 1$ and consider our first goal met if our implementation likewise finds $k = 8$ to perform best and our silhouette score is within $[0.4085, 0.4515]$.

Next we use the same *centers of performance* and evaluation criteria, silhouette score, to evaluate other clustering methods. We select three methods, standard K-means, and two methods that allow for non-spherical clusters, Gaussian Mixed Models, and Gustafson, Kessel, and Babuska Fuzzy K-means. For each method, we repeat the same investigation for $k = 1,\ldots,20$.

All analyses were performed using R Statistical Software v4.3.2 [16]. The public soccer-log data set was accessed with the jsonlite R package [12, 14]. Data manipulation and cleaning was accomplished with various packages from the tidyverse [22, 9]. PlayeRank soft-clustering silhouette calculations were done through extension of the silhouette function from the cluster R package [11]. K-means, Gaussian Mixed Models, and Gustafson, Kessel, and Babuska Fuzzy K-means were done with the stats, mclust, and fclust R packages, respectively [16, 19, 5]. Temporal data was handled with the clock R package [21]. Plots were created using ggplot2, ggsoccer, and ggdist [22, 20, 8, 7].

# Results

::: {#fig-si-comp}

![](/figs/p_silh.png)

Comparison of Silhouette Index by K for four methods
:::


::: {#fig-cluster_comp layout-ncol=2}

![Gustafson, Kessel, and Babuska Fuzzy K-means](/figs/p_gkb8.png){#fig-gkb8}

![PlayeRank K-means](/figs/p_prk8.png){#fig-prk8}

Comparison of Clustering Results by Method. (a) Roughly the same orientation of clusters as presented in PlayeRank [15], but allowance for non-spherical leads to better clusters. (b) Visually nearly identical to the PlayeRank results; light grey points are hybrid *centers of performance*.
:::

We were able to verify the results presented in PlayeRank by Pappalardo et al. [15]. Through our implementation of their algorithm, we found $k = 8$ clusters to have the best silhouette score and and the silhouette score was within our interval of $[0.4085, 0.4515]$ at $0.4151$. Of the other methods attempted, Gustafson, Kessel, and Babuska Fuzzy K-means also produced the best results in terms of silhouette score with $k = 8$ clusters. This method also had a higher silhouette score than the soft-clustering algorithm proposed in PlayeRank, at $0.5483$ indicating better performance. The other two methods performed best at different $k$ number of clusters but did not perform better than either the PlayeRank soft-clustering algorithm or Gustafson, Kessel, and Babuska Fuzzy K-means. The comparison across all $k = 1,\ldots,20$ number of clusters for the four methods is in @fig-si_comp.

A comparison of the clusters created by the top two methods, Gustafson, Kessel, and Babuska Fuzzy K-means and PlayeRank K-means is in @fig-cluster_comp. The random clusters were re-numbered to their closest match from the plot provided by Pappalardo et al. and to align the cluster colors for better visual comparison. The clustering present in @fig-prk8 closely matches that from PlayeRank and there are some noticeable differences in @fig-gkb8 [15].

# Discussion 

::: {#fig-cluster_comp layout-ncol=2}

![Small number of observations](/figs/p_sparse.png){#fig-sparse}

![Multiple areas of responsibility](/figs/p_maor.png){#fig-maor}


Potential Problems with Centers of Performance in PlayeRank
:::

We were able to achieve our goals for this project. After implementing the soft-clustering algorithm from PlayeRank, our investigation found nearly identical results to those reported by Pappalardo et al. [15]. Additionally, our use of Gustafson, Kessel, and Babuska Fuzzy K-means resulted in better clusters as evaluated by silhouette score. This is probably a result of the allowance for non-sphereical clusters, which is most evident on the wings, C1 and C4, in @fig-cluster_comp, and likely more accurately models the role of those players.

While we were able to achieve our goals, there are other areas of interest for possible improvement. The first few relate to the calculation of *centers of performance*. The first is the possible exclusion of players without a significant number of events. A player with few events in a match would likely indicate either a late substitution or an early one due to injury. In either event, a limited number of events limits the meaningfulness of the calculation as seen in @fig-sparse. The second is an exploration of another way to calculate the *center of performance*. The current method of calculation, a simple average position of all events, is unable to accurately account for players who switch area of responsibility within a match as seen in @fig-maor. Finally, the current methodology only takes into consideration spatial data of events and leaves out the additional temporal and contextual information available. The two events in @fig-sparse are annotated with this additional information, the event type and when the event took place in the match in terms of period and minute from start of the period. A method which includes this temporal and contextual information may provide a more accurate clustering.

An additional area that warrants further investigation is the consideration of alternative internal clustering validation metrics. In our investigation, we relied on the silhouette score as it was the metric with which Pappalardo et al. presented their findings [15]. The silhouette score is regarded as a dependable internal metric for clustering validation due to its independence from cluster diameter and lack of optimization for Gaussian contexts [10]. However, there is concern with internal clustering validation wherein the metrics used for evaluation are the same as those optimized by the clustering algorithms [13]. Contextual information, such as the assigned position of the player and team tactical formation, could potentially allow for the use of external clustering validation metrics in future research. 

# References

[1] Personal communication with Manager, Player Personnel Columbus Crew. Oct. 2023.
[2] Joel Brooks, M. Kerr, and J. Guttag. “Developing a Data-Driven Player Ranking in Soccer Using Predictive Model Weights”. KDD (2016). doi: 10.1145/2939672.2939695.
[3] Tom Decroos et al. “Actions Speak Louder than Goals: Valuing Player Actions in Soccer”. knowledge discovery and data mining (2019). doi: 10.1145/3292500.3330758.
[4] J. Duch, Joshua S Waitzman, and L. A. N. Amaral. “Quantifying the Performance of Individual Players in a Team Activity”. PLoS ONE (2010). doi: 10.1371/JOURNAL.PONE.0010937.
[5] M.B. Ferraro, P. Giordani, and A. Serafini. “fclust: An R Package for Fuzzy Clustering”. The R Journal 11 (2019). url: https://journal.r- project.org/archive/2019/RJ- 2019- 017/RJ- 2019- 017.pdf.
[6] Joachim Gudmundsson and Michael Horton. “Spatio-Temporal Analysis of Team Sports”. ACM Computing Surveys (2017). doi: 10.1145/3054132.
[7] Matthew Kay. ggdist: Visualizations of Distributions and Uncertainty. R package version 3.3.1. 2023. doi: 10.5281/zenodo. 3879620. url: https://mjskay.github.io/ggdist/.
[8] Matthew Kay. “ggdist: Visualizations of Distributions and Uncertainty in the Grammar of Graphics”. IEEE Transactions on Visualization and Computer Graphics (2024), pp. 1–11. doi: 10.1109/TVCG.2023.3327195.
[9] Max Kuhn and Hadley Wickham. Tidymodels: a collection of packages for modeling and machine learning using tidyverse principles. 2020. url: https://www.tidymodels.org.
[10] Jean-Charles Lamirel, Nicolas Dugué, and Pascal Cuxac. “New eﬀicient clustering quality indexes”. IEEE International Joint Conference on Neural Network (2016). doi: 10.1109/IJCNN.2016.7727669.
[11] Martin Maechler et al. cluster: Cluster Analysis Basics and Extensions. R package version 2.1.6 — For new features, see the ’NEWS’ and the ’Changelog’ file in the package source). 2023. url: https://CRAN.R-project.org/package=cluster. 
[12] Jeroen Ooms. “The jsonlite Package: A Practical and Consistent Mapping Between JSON Data and R Objects”. arXiv:1403.2805 [stat.CO] (2014). url: https://arxiv.org/abs/1403.2805.
[13] Julio-Omar Palacio-Niño and Fernando Berzal Galiano. “Evaluation Metrics for Unsupervised Learning Algorithms”. ArXiv (2019).
[14] Luca Pappalardo et al. “A public data set of spatio-temporal match events in soccer competitions.” Scientific Data (2019). doi: 10.1038/S41597-019-0247-7.
[15] Luca Pappalardo et al. “PlayeRank: Data-driven Performance Evaluation and Player Ranking in Soccer via a Machine Learning Approach”. ACM Transactions on Intelligent Systems and Technology (2019). doi: 10.1145/3343172.
[16] R Core Team. R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing. Vienna, Austria, 2023. url: https://www.R-project.org/.
[17] Peter J. Rousseeuw. “Silhouettes: A graphical aid to the interpretation and validation of cluster analysis”. Journal of Computational and Applied Mathematics 20 (1987), pp. 53–65. issn: 0377-0427. doi: https://doi.org/10.1016/0377- 0427(87)90125-7. url: https://www.sciencedirect.com/science/article/pii/0377042787901257.
[18] Oliver Schulte et al. “Apples-to-Apples: Clustering and Ranking NHL Players Using Location Information and Scoring Impact”. Proceedings of the MIT Sloan Sports Analytics Conference (2017).
[19] Luca Scrucca et al. Model-Based Clustering, Classification, and Density Estimation Using mclust in R. Chapman and Hall/CRC, 2023. isbn: 978-1032234953. doi: 10.1201/9781003277965. url: https://mclust-org.github.io/book/.
[20] Ben Torvaney. ggsoccer: Plot Soccer Event Data. R package version 0.1.7. 2022. url: https : / / CRAN . R - project . org / package=ggsoccer.
[21] Davis Vaughan. clock: Date-Time Types and Tools. R package version 0.7.0. 2023. url: https://CRAN.R-project.org/ package=clock.
[22] Hadley Wickham et al. “Welcome to the tidyverse”. Journal of Open Source Software 4.43 (2019), p. 1686. doi: 10.21105/ joss.01686.