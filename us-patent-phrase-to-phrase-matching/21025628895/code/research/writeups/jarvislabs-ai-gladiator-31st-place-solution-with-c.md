# 31st place solution [with code]

**Rank:** 31
**Author:** Atharva Ingle
**Collaborators:** Atharva Ingle
**Votes:** 44

---

Thank you organizers and Kaggle for organizing such a nice competition. Also, thanks to everyone for sharing during the competition. I learned a lot from discussions and some great notebooks.

I have open-sourced my code here: https://github.com/Gladiator07/U.S.-Patent-Phrase-to-Phrase-Matching-Kaggle

You can also view all my training logs on the Weights & Biases dashboard [here](https://wandb.ai/gladiator/USPPPM-Kaggle)

Final Inference Notebook [here](https://www.kaggle.com/code/atharvaingle/uspppm-inference-ensemble-hill-climbing)

Seeing the top solutions, my solution seems to be very simple 😅 and it basically relies on the diversity of models trained. 

# Tools used
- HuggingFace [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) and [datasets](https://huggingface.co/docs/datasets) for the complete code base
- [Hydra](https://hydra.cc) for configuration management
- [Weights & Biases](https://wandb.ai/site) for experiment tracking
- Git/GitHub for code tracking
- Google Cloud Bucket for storing models
- A100 for training large models and RTX 5000 for smaller models

This setup allowed me to utilize the limited time I had during the competition fully. I could run a series of experiments by just changing some flags from the command line itself.

I spent a lot of time at the start of the competition to have a reliable CV strategy. I tried all strategies shared in public notebooks and discussions and finally settled on grouped by anchor and stratify on score strategy as follows:
```    
    train_df["score_bin"] = pd.cut(train_df["score"], bins=5, labels=False)
    train_df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = sgkf.split(
        X=train_df,
        y=train_df["score_bin"].to_numpy(),
        groups=train_df["anchor"].to_numpy(),
    )
    for fold, (trn_idx, val_idx) in enumerate(folds):
        train_df.loc[val_idx, "fold"] = fold
    train_df["fold"] = train_df["fold"].astype(int)
```

# Summary
- Used `[s]` instead of `[SEP]`. Improved CV of baseline model from `0.81408` to `0.81906`
- Trained most of the models with three losses for diversity: MSE, BCE, and Pearson loss (Pearson worked best for me)
- Used multi-sample dropout for some models.
- Tried various types of pooling heads for diversity:
    - Attention Pooling
    - Mean Pooling
    - Max Pooling
    - Mean Max Concatenate Pooling
    - Conv1D Pooling
- Used low learning rates for larger models
- Dynamic Padding and Uniform Length Batching for increasing training and inference speed

# Ensemble
I used the hill-climbing approach explained by Chris Deotte [here](https://www.kaggle.com/code/cdeotte/forward-selection-oof-ensemble-0-942-private/notebook). I used a tolerance of 0.0 for the final submission (i.e add a new model only if increases the CV by some tolerance, in this case, it added the models as long as the CV increases).
Also, I scaled all model's predictions by `MinMaxScaler` for ensembling.
However, I also selected a submission with a tolerance of 0.0003 to not overfit on CV but it turned out that the submission with the tolerance of 0 scored the highest on private LB.

My final submission:
| Submission | CV Score | Public LB | Private LB |
| --- | --- |--- |--- |
|31 experiments (31 * 5 = 155 models) | 0.85484 | 0.8505 | 0.8652
| 8 experiments (8 * 5 = 40 models) | 0.85382 | 0.8504 | 0.8650

Only 8 experiments ensemble would have also gotten me the same rank 😂

# Things that didn't work for me
- Ordinal Regression.
- Posing the problem as classification and using cross-entropy loss / weighted cross-entropy loss.
- I really wanted to try the SVR trick inspired by @cdeotte and @titericz from PetFinder comp but couldn't make it work as I started implementing it in the last 2 days of the competition and there was not enough time to debug what went wrong.
- Simple average ensemble of models worked worse compared to a weighted ensemble by hill climbing.
- Second stage model stacking performed worse than hill climbing.
- I see many top teams benefitting from AWP. I will give it a shot in the next NLP comp :)
 
Here are the CV scores for final ensemble submission

| Experiment | CV Score |
| --- | --- |
| 107_microsoft-deberta-v3-large_fin-val-strategy-pearson-baseline | 0.8332 |
| 109_microsoft-deberta-v3-large_pearson-attention-pool | 0.8329 |
| 110_microsoft-deberta-v3-large_mse-msd | 0.8348 |
| 111_microsoft-deberta-v3-large_pearson-ms | 0.8342 |
| 112_microsoft-deberta-v3-large_mse-transformer-head | 0.8130 |
| 121_anferico-bert-for-patents_mse-baseline-low-bs | 0.8225 |
| 122_anferico-bert-for-patents_pearson-baseline-low-bs | 0.8223 |
| 123_anferico-bert-for-patents_mse-msd-low-bs | 0.8229 |
| 124_anferico-bert-for-patents_pearson-msd-low-bs | 0.8209 |
| 126_anferico-bert-for-patents_pearson-attention-pool-low-bs | 0.8212 |
| 128_microsoft-deberta-v3-large_pearson-mean-pool | 0.8333 |
| 129_microsoft-deberta-v3-large_mse-conv1d-pool | 0.8312 |
| 130_microsoft-deberta-v3-large_pearson-conv1d-pool | 0.8322 |
| 134_anferico-bert-for-patents_pearson-conv1d-pool | 0.8200 |
| 140_microsoft-deberta-v3-large_pearson-mean-max-concatenate-pool | 0.8342 |
| 144_anferico-bert-for-patents_pearson-mean-max-concatenate-pool | 0.8215 |
| 152_microsoft-deberta-v2-xlarge_pearson-lowlr | 0.8270 |
| 154_microsoft-deberta-xlarge_pearson | 0.8263 |
| 164_albert-xxlarge-v2_mse | 0.8108 |
| 165_albert-xxlarge-v2_bce | 0.8109 |
| 166_albert-xxlarge-v2_pearson | 0.8110 |
| 168_google-electra-large-discriminator_bce | 0.8143 |
| 169_google-electra-large-discriminator_pearson | 0.8098 |
| 172_funnel-transformer-large_pearson | 0.8219 |
| 174_funnel-transformer-xlarge_bce | 0.8238 |
| 175_funnel-transformer-xlarge_pearson | 0.8225 |
| 177_albert-xxlarge-v2_bce-lowlr | 0.8102 |
| 183_microsoft-deberta-large_bce | 0.8214 |
| 197_microsoft-deberta-v2-xlarge_mse-pearson | 0.8267 |
| 205_microsoft-cocolm-large_pearson-msd | 0.8218 |
| 208_microsoft-cocolm-large_mse-conv1d-pool | 0.8180 |


# Acknowledgments
I would like to thank everyone who shared during the competition. I learned a lot and will try to apply all the learning in the next competition. Also, a special thanks to [this](https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently) amazing notebook by @rhtsingh, got to learn a lot from this. And the legend @cdeotte for his detailed [discussion](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175614) and notebook on hill climbing.

**Also, a huge thank you to [jarvislabs.ai](https://jarvislabs.ai) for the GPU support. The platform enabled me to do multiple experiments rapidly with instant and powerful GPU instances. All my models were trained on [jarvislabs.ai](https://jarvislabs.ai) and this could not have been achieved without them.**