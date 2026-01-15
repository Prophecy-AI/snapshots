# ３rd Place - Diversity and Hill Climbing

**Rank:** 3
**Author:** nice kazusan
**Collaborators:** nice kazusan
**Votes:** 12

---

First of all, I would like to express my sincere thanks to the organizers and participants of this competition, which was a valuable learning experience.
Also, congratulations to @cdeotte for winning 1st place twice in a row and  @mahoganybuttstrings for winning 2nd place.
I’m especially grateful to @cdeotte for the helpful information he provides us every time.

## Outline
### Step1
Now, in this competition, I was troubled by unstable CV-LB correlations and ignored public LB for consideration.
The original competition dataset was initially considered for use as part of the training data.
However, I could not improve the CV as much as I would have liked and decided not to adopt it.
The first work considered was a Meta-model combining a total of 18 models (5kfold) of Catboost, LGBM, XGB, NN and AutoGluon.
For these 18 models, I created multiple conditions using AutoFeat and Optuna to focus on model diversity.
CV score was 0.05893.

### Step2
In particular, I referred to [this notebook](https://www.kaggle.com/code/onurkoc83/catboost-xgboost-with-new-features)  and made additional considerations regarding the features.(thanks @onurkoc83)
And finally, I made an ensemble using Hill Climbing.
In addition to the Meta-model above, the Hill Climbing selected the following six models at this time.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9309722%2F9b2185b4ce10e1d9445269191255a153%2F1.png?generation=1748930521779538&alt=media)
This improved the CV to 0.05885.
And with this model, which had the best CV, I was able to come in third.

## Concluding thoughts
I was fortunate enough to take 3rd place this time, but I’m still lacking in knowledge and will continue to learn more.
Thank you for your continued guidance.
