# 5th place solution & magic

**Rank:** 5
**Author:** tkm2261
**Collaborators:** guchio3, kenmatsu4, Yiemon773, KF, tkm2261
**Votes:** 67

---

Hi, All
First of all, congrats winners! We were a little bit behind you guys. We enjoyed this competition very much. Due to the magic, this competition was a little bit different from standard ML competitions. It makes us feel solving some kind of puzzle although It was also fun for us

# The Magic

I guess that this is just a bug introduced when they created this task. Here shows a representative example.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F436401%2F1bcbc6606aa7a3e112698c5ddfac12ad%2FScreen%20Shot%202020-06-16%20at%205.37.10%20PM.png?generation=1592354249239961&amp;alt=media)

The given original annotation is “onna” but it is too weird. The true annotation should be “miss” (this is a negative sentence). We think that the host applied a wrong slice obtained on the normalized text without consequence spaces for the original text with plenty of spaces, emojis, or emoticons. Thus, this competition pipeline should be as follows.

* Recover true annotation from the buggy annotation (pre-processing).
* Train model with true annotation.
* Predict the right annotation.
* Project back the right annotation to the buggy annotation (post-processing).

We call the pre-processing and post-processing as magic. After we found that, our score jumped from 0.713 to 0.721. Maybe, we can also do the 4 steps with an end-2-end model as some people claimed that they did not use any postprocessing. From 0.721 to 0.731, we improved individual models, ensembled models, and improved the pre- and post-processing.

# Model

* We use RoBERTa and BERTweet.
* Ensembled all 5 member’s models in the char-level.
* We do not do special things in our model training.
* We only use train.csv. (no sentiment140 and complete-tweet-sentiment-extraction-data)

# Post-process improvement

Assuming the model (token level) is perfect, we maximize the Jaccard score with the pre- and post-process. This is an example.
https://www.kaggle.com/tkm2261/pre-postprosessing-guc

```
&gt;&gt;&gt;&gt; FOLD Jaccard all = 0.9798682269792468
&gt;&gt;&gt;&gt; FOLD Jaccard neutral = 0.9961648726550028
&gt;&gt;&gt;&gt; FOLD Jaccard positive = 0.969571077575057
&gt;&gt;&gt;&gt; FOLD Jaccard negative = 0.96793968688976
```

Under the perfect model assumption, this result can be interpreted that we can achieve 0.9798682269792468 if the model is perfect. Then, we apply this postprocessing for our model prediction. This pipeline worked pretty well. Any members can test their own postprocessing idea and share it with team if it improves the score.

In summary, to our knowledge, this competition is a competition that reproduces embedded human errors. When I found the magic, I am also disappointed a little bit. But, maybe, noticing such bugs in data should be one skill of DS. We should not just apply model but dive into data carefully.

If you have any questions, plz feel free to post it in this thread.

Thank you.