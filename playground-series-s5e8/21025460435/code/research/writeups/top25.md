# 🏆 Solution Write-up – Top 25

**Rank:** 25
**Author:** abdelbasset ben kerrouche
**Collaborators:** abdelbasset ben kerrouche
**Votes:** 12

---

***1. Introduction***

The competition was a binary classification task using a bank marketing dataset.

**Goal**: Predict whether a client will subscribe to a bank term deposit.

**Data size**: Training set ~750,000 rows × 16 features, Test set ~250,000 rows × 17 features.

**Result**: I achieved Top 25 on the private leaderboard.

At the beginning, my LB score was around 0.97555. After extensive feature engineering, data augmentation, and model experimentation, I was able to significantly improve my results and reach the final placement.

**2. Feature Engineering & Data Augmentation**
Initially, I created around 30 new features manually.

Later, I discovered a very helpful notebook by [Chris Deotte](https://www.kaggle.com/cdeotte), which inspired me to engineer more complex transformations thank you chris im alway teaching from of you.

Using his ideas, I created more than 447 new features, boosting my LB to 0.97717 , my CV 0.976.

Since the dataset was imbalanced (87% class 1, 13% class 0), I applied synthetic data generation ( synthcity ) to oversample the minority class. This improved model robustness and reduced bias toward the majority class but this didn't work for me.

**3. Models & Training Strategy**

I used multiple models and several rounds of training:

Base models:

LightGBM (LGBM)

XGBoost (XGB)

Neural Networks (NN)

CatBoost & other tree-based models (in some experiments)

Tuning strategy:

For XGB, I trained multiple times with different learning rates: 0.1 → 0.07 → 0.02.

For LGBM, I trained 5 times with different random seeds.

For NN, I trained 4 runs with different seeds.

**Cross-validation (CV):**

I experimented with different folds:

5-fold,

7-fold,

11-fold, which turned out to be the strongest and most stable.

I always trusted my CV results more than the LB, which helped me avoid leaderboard overfitting and guided my model selection, the best CV was Logistic Regression AUC: 0.976979405561156

**First approach (simple ensemble):**

Started with only 2 models (LGBM + XGB).

The score improved after retraining with adjusted hyperparameters .

Second notebook (larger ensemble):

Trained 13 models (mix of LGBM, XGB, CatBoost, NN, Trees).

Used 3 meta-models (NN, Tree, XGB) for stacking.

Trained 4 models on residuals (error-focused).

Finally added a logistic regression (LR) blender.

This achieved CV 0.9758 and  0.97683 LB.

4. Final Ensemble (Blending & Stacking)

Combined predictions from:

5 LGBMs

3 XGBs

5 NNs

13 diverse models from the second notebook

All stacked with Logistic Regression (LR).

I also applied weighted blending:

My strongest notebook result was multiplied by 0.3.

Other weaker models were assigned smaller weights.

In total, around 10 prediction sets were blended in the final solution.

**5. Challenges**

**Large dataset:** Training was computationally expensive, so I split the workflow into 8 separate notebooks:

4 notebooks dedicated to data preparation and saving processed files, i make 3 huge data one for NN and one for LGBM and  the other for XGB

4 notebooks for model training, stacking, and blending , 2 for blrnding 

Time management: Careful scheduling was required, running multiple seeds and long experiments overnight.

**6. Results**

Public LB: 0.97733

Private LB: Top 25 🎉

**7. Key Takeaways**

Feature engineering was the biggest performance booster (+447 new features were crucial).

11-fold CV gave more stability and stronger results compared to 5 or 7 folds.

Always trusting CV over LB kept my solution robust and safe from leaderboard overfitting.

Model diversity (NN + LGBM + XGB + Trees) reduced variance.

Multiple seeds + residual models improved robustness.

Weighted blending was safer than relying on a single strong model.

Splitting the pipeline into separate notebooks saved time and memory , chris idea [here](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/601015)

**8. Acknowledgments**

Special thanks to Chris ,everyday learn from him  specialy the feature engineering ideas that inspired a big part of my solution .