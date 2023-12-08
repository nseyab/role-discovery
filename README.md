# Introduction

Data-driven performance evaluation of players is becoming more widely used in the soccer industry\cite{developing_brooks_2016,actions_decroos_2019, quantifying_duch_2010, playerank_pappalardo_2019}. For a team, the ability to compare players is critical to success as it provides decision support to many areas, i.e. which athletes to include on match day rosters, which to start and who to substitute, who to target for roster building during the transfer windows for professionals or who to recruit for collegiate teams\cite{crew_2023}. As such, there is great interest in the creation of a data-driven performance rating metric that quantifies the quality of a player's performance based on related spatio-temporal events during a match. The rating can then be used for a direct comparison of players. 

One such rating, PlayeRank, proposed by Pappalardo et al., was groundbreaking in three areas: it was the first multi-dimensional rating approach; the first to considered the specificity of each player's role on the field; and the first to evaluate the goodness of ranking in a quantitative manner similar to recommender systems in Information Retrieval. The authors offer several areas for potential improvement, one specifically mentioned is development of improved methods for identifying a player's role within a match or a specific portion of it. Due to the modular nature of PlayeRank, an other methods of role detection can be seamlessly integrated allowing users to personalize role detection based on their individual requirements.\cite{playerank_pappalardo_2019}.

In this paper we will follow in the footsteps of Pappalardo et al. with regard to the player role detection. We will attempt to verify and potentially improve upon their results through the use of unsupervised machine learning methods to identify a set of player roles utilizing match events in a soccer-logs dataset. 

# Related Works
The advent of extensive data sets depicting soccer performance has greatly contributed to the recent progress in soccer analytics. Soccer-logs, comprehensive records of all events taking place during a match, have emerged as a prevalent data format, enabling analysis of various aspects of soccer at both team and individual levels\cite{public_pappalardo_2019, playerank_pappalardo_2019}. One aspect of PlayeRank that is of particular interest is role detection\cite{playerank_pappalardo_2019}.

### Performance Metrics and Evaluation 
Prior to PlayeRank, there were various approaches to evaluating player performance. One of the pioneering efforts was the flow centrality (FC) metric, a network approach that quantifies individual player contributions and overall team performance. It provided a comprehensive evaluation of player performance and its impact on team outcomes\cite{quantifying_duch_2010}. Another interesting metric was the Pass Shot Value (PSV) metric, which focuses on the value of completed passes and utilizes a supervised machine learning model to analyze pass locations and their relationship to generating shot opportunities\cite{developing_brooks_2016}. However, both of these metrics struggled to be adequate when applied outside of attacking players. PlayeRank, follows those earlier attempts, and sought to overcome their shortcomings. It is a multi-dimensional, data-driven framework and provides role-aware evaluation of player performance\cite{playerank_pappalardo_2019}.

### Player Role Recognition through Spatio-Temporal Events
One of the problems that plagued early attempts at data-driven player evaluation was lack of role awareness. Early attempts did not distinguish between where players were located on the pitch and what tasks they were assigned within the team's tactical structure. These approach shortcomings were noted by their developers\cite{developing_brooks_2016, quantifying_duch_2010}. PlayeRank was the first metric which proposed a method for producing a role-aware player evaluation\cite{playerank_pappalardo_2019}. Outside of the sport of soccer, other methods existed for role detection via spatio-temporal events\cite{spatiotemporal_gudmundsson_2017, schulte_2017}. However, these techniques were found inadequate when applied to soccer due to its lower event density \cite{playerank_pappalardo_2019}.   

# Methodology
This project aims to reproduce the role detection results found by Pappalardo et al. \cite{playerank_pappalardo_2019}. In order to meaningfully compare and verify those findings, we adopt the same overall average silhouette width $\bar{s}_{k}$, or silhouette score, to validate our methods. This is the mean of $s_{i}$ for all $i$ in the data set, with $s_i$ defined as: $$s_{i} =\frac{b_i - a_i}{\textsf{max}(a_i, b_i)}$$ where, $a_i$ is the mean intra-cluster dissimilarity, and $b_i$ is the mean nearest-cluster dissimilarity \cite{rousseeuw_1987}.

We make use of a public data set of soccer-logs from Wyscout consisting of 3,251,294 events from 1,941 matches and 4,299 players. This data set includes the 2017/18 seasons of the first division competitions in England, France, Germany, Italy, and Spain, as well as the 2018 Fédération Internationale de Football Association (FIFA) World Cup and the 2016  Union of European Football Associations (UEFA) European Football Championship \cite{public_pappalardo_2019}. Each event record consists of twelve items describing seven characteristics, of which the following four were of primary interest in our study:
\begin{enumerate}
  \item eventId, eventName, subEventId, subEventName, tags: type of event and additional information
  \item matchId: identifier for the match
  \item playerId: identifier for player who generated the event
  \item positions: $(x, y)$ coordinate pair origin and destination positions of the event $(x,y) \in [0, 100]$ ($x$ and $y$ indicate event's nearness to opponent's goal and nearness to right side of the pitch, respectively, in percentage)
\end{enumerate}
We omit goalkeeping events and goalkeepers from the analysis as the role has different rules from other players. 

:::{#fig-events}

![](/figs/p_events.png)

Distribution of Events by Player-Match Group
:::

In order to achieve our goal, we implement, in R, the same method introduced by Pappalardo et al. \cite{playerank_pappalardo_2019}. This method calculates a *center of performance* for each player in each match by average position $(\bar{x}, \bar{y})$ of events generated by the player $u$ in a given match $m$, such that a *center of performance* for a given player and match is denoted $c_u^m = (\bar{x}, \bar{y})$. A *soft clustering* is then applied by k-means over the *centers of performance* of all players in all matches. The *soft clustering* allows for "hybrid" roles, where the *center of performance* is assigned to two or more clusters based on a tolerance threshold, $\delta_{s}$. 

This is accomplished by calculating the silhouette width in reference to all clusters such that for any *center of performance*, $i$, in cluster $C_i$ the silhouette for each cluster $C_k$, where $C_k \ne C_i$, is calculated: 
$$ s_{iC_k} = \frac{d_i - a_i}{\textsf{max}(a_i, d_i)} $$ 
where $d_i$ is the mean nearest-cluster dissimilarity to cluster $C_k$. $i$ is assigned to all clusters $C_k$ where $s_{iC_k} \in (0, \delta_{s})$. The silhouette widths are calculated again using the standard method. For hybrid $i$, a average silhouette width is calculated as $\sum_{j=1}^n{s_{ij}} / n$, where $n$ is the total number of clusters $i$ is a member. Papplardo et al. varied $k = 1,...,20$ with $\delta_{s} = 0.1$, or $5%$ of centers classified as "hybrid". They found that $k = 8$ performed best with a silhouette score of $0.43$. We will omit $k = 1$ and consider our first goal met if our implementation likewise finds $k = 8$ to perform best and our silhouette score is within $[0.4085, 0.4515]$.

After removing events originating with goalkeepers and calculating *centers of performance*, we had 2,873,772 events which translated into a total of 49,495 *centers of performance*. The distribution of this data is shown in @fig-events. 

# Results

::: {#fig-si-comp}

![](/figs/p_silh.png)

Comparison of Silhouette Index by K for four methods
:::




We proceeded first with an investigation of normal K-means for $K = 1,...,20$. Results from a comparison of the Silhouette Index for each K, revealed the best to be $K = 2, 5, 8$, respectively. The results of this investigation are summarized in @fig-si-comp. PlayeRank \cite{playerank_pappalardo_2019}.

::: {#fig-cluster_comp layout-nrow=2}

![Gustafson, Kessel, and Babuska Fuzzy K-means](/figs/p_gkb8.png){#fig-gkb8}

![PlayeRank K-means](/figs/p_prk8.png){#fig-prk8}

![K-means](/figs/p_km8.png){#fig-km8}

![Gaussian Mixed Models](/figs/p_gm9.png){#fig-gm9}

Comparison of Clustering Results by Method
:::

Implementation of the PlayeRank *soft-clustering* presented some difficulties. Initial algorithms were successfully applied to a subset of the data, however, these failed due to memory issues when applied to the entire dataset. We are close to a solution that works but the time and memory requirements for computation of silhouettes for all *centers of performance* has slowed progress. We were nearly there at least for $k = 8$ tonight, but the R session crashed while running the function for the final calculation of silhouettes. The current results are not yet acceptable, but we are hopeful that over the next few days, the current issues will be resolved and we will be able to verify or dispute the results from PlayeRank.

# Discussion

There are three areas of interest for possible improvement. The first two relate to the calculation of *centers of performance*. The first is the possible exclusion of players without a significant number of events. The mean of events by player by match is $\mu = 73$ with a standard deviation of $\sigma = 26.5$, as seen in @fig-events. Players with less than $\mu - 2\sigma$ events in a match would most likely indicate either a late substitution or an early one due to injury. In either event, a limited number of events would likely limit the meaningfulness of the calculation as seen in @fig-sparse. The second is an exploration of another way to calculate the *center of performance*. The current method of calculation is a simple average position of all events, this type of calculation of *centers of performance* is unable to accurately account for players who switch area of responsibility within a match as seen in @fig-maor. Finally, the current methodology only takes in to consideration where on the pitch events took place and disregards types of events and their outcomes. A method which includes more information may provide a more accurate clustering. 

::: {#fig-cluster_comp layout-ncol=2}

![Small number of observations](/figs/p_sparse.png){#fig-sparse}

![Multiple areas of responsibility](/figs/p_maor.png){#fig-maor}


Potential Problems with Centers of Performance in PlayeRank
:::

Another area of improvement is looking into other Clustering Validation Index (CVI). We evaulated our methods based on the Silhouette Index, because that is the metric with which Pappalardo et al. presented their findings \cite{playerank_pappalardo_2019}. However, there exist numerous other internal CVI that be used and potentially verify further that $k = 8$ is the appropriate number of roles among field players on the pitch. 
