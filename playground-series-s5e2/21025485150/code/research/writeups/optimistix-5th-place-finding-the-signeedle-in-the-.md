# 5th place: Finding the SigNeedle in the NoiseStack

**Rank:** 5
**Author:** Optimistix
**Collaborators:** Optimistix
**Votes:** 14

---

When this playground episode started, I was expecting a classification problem, and was a bit surprised to see that we had another regression problem. Moreover, it became clear within a day or two that none of the features seemed related to the target, and the original data didn't seem to be "real world" at all. Several Kagglers pointed this out to Kaggle admin, and many of us were hoping for a 3 week competition with a new dataset. Instead, Kaggle responded with 12 times as much training data we had. While this motivated some - notably @cdeotte - to exploit the synthetic data generation process to extract some signal, others including me were significantly demotivated.

Perhaps the most consequential thing I did in this month's competition was to ask the creator of the original dataset (@souradippal) how he created the dataset, and when he graciously shared the code along with a description, it became clear that the target was indeed sampled at random. However, the details of the process again created scope for extracting some more signal, which @cdeotte skillfully exploited. 

For most of this month, I wasn't very interested in working on this competition, and just ran a few scripts and tried some blends. In contrast to most months, I hardly ran any code on my own laptop, just using Kaggle notebooks. I didn't even start building an ensemble until there were just 5 days to go. But the posts and scripts generously provided by @cdeotte & others finally motivated me to try for the last few days, and I finally started building my own ensembles. I should thank the following for sharing code and insights:

@cdeotte, @masayakawamata, @mikhailnaumov, @vktsyp, @souradippal, @vyacheslavbolotin, @yunsuxiaozi, @rtenorioramirez, @ravi20076, @siukeitin, @paddykb 

In the end, I had an ensemble of 25 models, with CV = 38.59273 (using Autogluon). This was the best CV I achieved, but the LB was 38.85316. The difference of over 0.26 incidicated that I might be overfitting the training data, since most other solutions only had a difference of about 0.2 between CV and LB. So I preferred the ensemble with the best LB score while maintaining LB - CV ~ 0.2, which had CV = 38.63914 and LB = 38.83889 (using Ridge Regression). As a standalone final submission, this would have scored 38.64268, and netted me 18th place. However, I couldn't resist blending my ensembles with @mikhailnaumov's top scoring public notebook - this yielded a slightly worse public LB score of 38.8224, but I chose it because I knew such blends often do well on the private LB, and indeed, the private LB score was 38.63455. So I ended up at no. 5. I almost feel like I owe an apology to @mikhailnaumov for using his solution to leapfrog him 😀

Along the way, I did try to build on some of the solutions by @cdeotte and others, and did so to some extent, but also kept running into GPU & memory issues. All in all, it was an odd month, where I wasn't motivated to work on the competition for the most part, yet ended up at no. 5, albeit without the satisfaction of earlier Top 10 finishes where I'd put in the thought and effort on a consistent basis. Let me finish by congratulating @cdeotte, @ravi20076, @mikhailnaumov, @masayakawamata , @swagician and all others who finished at the top, and wish everyone all the best for March - Happy Kaggling!