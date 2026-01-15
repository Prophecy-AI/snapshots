# 327th Place Solution Write-Up (top 10%)

**Rank:** 327
**Author:** Meagan Burkhart
**Collaborators:** Meagan Burkhart
**Votes:** 1

---

## Context
**Business Context:** https://www.kaggle.com/competitions/playground-series-s4e2/overview
**Data Context:** https://www.kaggle.com/competitions/playground-series-s4e2/data

**My Notebook:** https://www.kaggle.com/code/mvoulo/predictingobesity
## Overview of My Approach
Even though I knew this was synthetic data and that there might not be a reason to look at exisiting literature related to obesity, I decided to treat this as a "real" project. I did my research. 


## Details of the Submission 
### Exisiting Literature 
I found several articles that I found useful but the most interesting one that I utilized for my solution was: https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2805328

The focus was on creating a "Healthy Lifestyle Score"
```
 #HealthylifestyleFactors
# source https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2805328
df_train_merge['Nosmoke']=np.where(df_train_merge['SMOKE_yes']==0, 1, 0)
df_test_merge['Nosmoke']=np.where(df_test_merge['SMOKE_yes']==0, 1, 0)
df_train_merge['exercise']=np.where(df_train_merge['FAF']>=1, 1, 0)
df_test_merge['exercise']=np.where(df_test_merge['FAF']>=1, 1, 0)
df_train_merge['alco']=np.where((df_train_merge['CALC_Sometimes']==1)|(df_train_merge['CALC_no']==1), 1, 0)
df_test_merge['alco']=np.where((df_test_merge['CALC_Sometimes']==1)| (df_test_merge['CALC_no']==1), 1, 0)
df_train_merge['HealthyDiet']=np.where((df_train_merge['FAVC_yes']==1) &(df_train_merge['CH2O']>=2), 1, 0 )
df_test_merge['HealthyDiet']=np.where((df_test_merge['FAVC_yes']==1) &(df_test_merge['CH2O']>=2), 1, 0 )

```

Of course, the competition data did not include every variable I needed to exactly recreate the Healthy Lifestyle Score used in the research, but I decided to focus on the core concepts:
not smoking, exercising regularly, no or moderate alcohol consumption, and eating a healthy diet. 

### Feature Engineering
 My final features included in my models were:

``` 
X=df_train_merge[['BMI', 'Gender01','Age', 'Height', 'Weight','FCVC', 'NCP', 
       'family_history_with_overweight_yes', 'FAVC_yes',
       'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no', 'SMOKE_yes',
       'SCC_yes', 'CALC_Sometimes', 'CALC_no', 'MTRANS_Bike',
       'MTRANS_Motorbike', 'MTRANS_Public_Transportation','FAF', 'TUE','CH2O',
       'MTRANS_Walking', 'HealthyLifestyle_score']]
```

### Model Comparisons
I compared 7 baseline models before I selected the top 3 and tuned them with optuna. This is the boxplot that shows my model comparison:
![modelcomp](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3113438%2F3ee579a2145bd3b689fb1f2dce2f6287%2Fmodelcomp_obesity.png?generation=1709277740950371&alt=media)

I ended up choosing the LGBM, RF, and XG models for my final voting classifier.

``` vc = VotingClassifier([
                       ('LGBM', LGBMClassifier(**lgb_params, verbosity=-1)),
                       ('RF', RandomForestClassifier()),
                       ('XG', xgb.XGBClassifier(**xg_params) )], voting='soft')
```

### Validation
I used 5-fold cross validation. I think that if I could do this again I would increase to 10-folds, but this worked out ok.

My final CV score was:
 0.9103 (+- 0.0046)

### Features Importance

![feature importance](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3113438%2Fb9577c625fc300a5fdb345871a1294e7%2Ffeatureimportance_obesity.png?generation=1709278342597964&alt=media)


### What I'd Do Differently
- optimize the VC weights: I just did equal weights (mostly out of laziness). I should have at least done a grid search to optimize the weights
- spend more time on EDA: I didn't do too much data visualizations because I didn't dedicate as much time to this competition as I initially intended


## Sources
https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2805328
Feature importance function for VC: https://towardsdatascience.com/custom-implementation-of-feature-importance-for-your-voting-classifier-model-859b573ce0e0

## Conclusion
Given that I didn't put as much time into this competition as I originally planned, I was pleased with the results of my notebook. Obviously there were things I could have done better/differently, but I am excited to share what I found related to the Healthy Lifestyle Score.