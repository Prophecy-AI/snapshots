# 56th solution of Our First silver medal !🥈 Just tune hyperparameters and models on the baseline.

**Rank:** 56
**Author:** medicine-wave
**Collaborators:** Buying Niu, medicine-wave
**Votes:** 36

---

# To record our easy ordinary solution which brings a not bad result！😊

Thanks for  the [training notebook](https://www.kaggle.com/code/yasufuminakama/pppm-deberta-v3-large-baseline-w-w-b-train) and  [inference notebook](https://www.kaggle.com/code/yasufuminakama/pppm-deberta-v3-large-baseline-inference) baseline of @Y.Nakama, We learned a lot.

## 【Data format】

Just like baseline: ['anchor'] + '[SEP]' + ['target'] + '[SEP]'  + ['context_text'] 

## 【Cross Validation】

We use the  `train_fold5.csv` and  `train_fold4.csv` from https://www.kaggle.com/datasets/helloggfss/foldsdump

## 【Ensemble】
| Model | Public LB | Private LB |
| --- | --- |
| bert_for_patent_5folds | 0.8312 |  0.8420 | 
| bert_for_patent_4folds | 0.8320 |  0.8414 | 
|electra_5folds| 0.8371|0.8504|
|electra_5folds_tuned|0.8383 | 0.8508 | 
|deberta_v3_large|0.8376|0.8510|
|deberta_v3_large|0.8380|0.8490|
|Funnel_xlarge_5folds|0.8380|0.8488|
|Funnel_large_4folds|0.8325|0.8416|
|**Average Ensemble 8 models**| **0.8533（33th）**| **0.8639（56th）**|

## 【Tune tricks】
**Retrain:**
- Firstly, train the single model about 6 epochs with learning rate about 2x10-5, and save it.
- Secondly, load the saved model state and train it again with a smaller learning rate like 5x10-6 about 5epochs.
- I guess it can help achieve the local minimum more easily, which always improve 0.002+ on CV or LB.

**Others:**
- Just tune warm up, learning rate and num of cycles.
- Adding attention head and Layernorm improve 0.002+ on the models except deberta.

## 【At last】
Thanks for my teamate @buyingniu, we tune the parameters all the days.😂
And I surely learn a lot new things from the top solutions! 💪