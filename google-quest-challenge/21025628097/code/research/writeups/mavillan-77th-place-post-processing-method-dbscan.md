# 77th place post-processing method: DBSCAN

**Rank:** 77
**Author:** mavillan
**Collaborators:** mavillan
**Votes:** 13

---

Thanks to all of you how shared your work and ideas during the competition!

I started this competition late and without any knowledge about BERT or the current transformers architectures. My goal was just to learn about BERT, so I'm very happy to be in the silver zone. 

## Base models
My base models were just variants of what was shared in the public kernels:

1. `bert-large` trained over `question_titile` + `question_body` + `answer`.
2. Two `bert-base` trained over `question_title`+`question_body` and `answer`, to predict answer and question targets separately.
3. Three `bert-base` trained over `question_title`+`question_body`, `question_body`+`answer` and `answer`,  to predict all targets jointly. 

I used 5-GroupKFold grouped by `question_title`. For the training process I first trained the regression head with BERT fixed, and then trained the whole network for 3 epochs.

My final submission was just the average of these three models.

## Post-processing

Two days previous to the competition end, I was at ~800th position in public leaderboard with ~0.39 score with no chances to enter the medal zone. Thanks to @corochann in this [discussion](https://www.kaggle.com/c/google-quest-challenge/discussion/118724) I found out why I wasn't able to go up in the leaderboard: Even when my models were pretty good at predicting scores, the only thing that matter for the evaluation metric is the order. **The solution**: find groups of target predictions which are very close and set them with the same value, i.e.,  **cluster** nearby target predictions, and [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) is pretty good at this task.

&gt; DBSCAN is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).

This algorithm requires two parameters: `eps` the maximum distance for two points to be considered neighbors (and be assigned to the same cluster), and `min_samples`  the number of samples in a neighborhood for a point to be considered as a core point.

Thus, my post-processing routine was:

```python
def cluster_predictions(submission, q_value=0.95):
	submission = submission.copy(deep=True)
	for column in submission.columns[1:]:
		print(column)
		max_dist = (submission[column]
				    .sort_values()
				    .diff()
				    .quantile(q_value))
		print(max_dist)
		if max_dist == 0: continue
		X = submission[column].values.reshape((-1,1))
		clustering = DBSCAN(eps=max_dist, min_samples=2).fit(X)
		print(np.unique(clustering.labels_))
		for cluster_id in np.unique(clustering.labels_):
			if cluster_id == -1: continue
			cluster_idx = np.argwhere(clustering.labels_ == cluster_id)
			cluster_median = np.median(submission.loc[cluster_idx, column])
			submission.loc[cluster_idx, column] = cluster_median
	return submission
```
for each target column, I compute the differences between the sorted predictions, and set `eps` as a high quantile of these differences. I tuned `q_value` through the 5-GroupKFold predictions and found 0.95 to be a good value. The post-processing gave me a boost of `0.39X -&gt; 0.42X` in public LB and `0.37X -&gt; 0.39X` in private LB. 

With more time and a better ensemble, I think this approach could have achieved a much better correlation in the private leaderboard. 

