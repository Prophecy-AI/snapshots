# [8th] Place Solution for the [Predict Calorie Expenditure] Competition

**Rank:** 8
**Author:** pinoystat
**Collaborators:** pinoystat
**Votes:** 10

---

The last submission I made in this compeition was 16 days ago. Hence, I was surprised that I made it in the 8th place.  My solution is a simple ensemble with weights determined by Hill Climbing.  The final ensemble is composed of 5 test predictions from 5 notebooks. The models selected by the algo are 3 Tensorflow , 1 Catboost and 1 XGBoost.  In this competition, I mainly focused in building Tensorflow models to gain deeper understanding (by doing) about tensors, tensor manipulations and this library. 

**Notebooks Ensemble of Predictions Summary**
Notebook # F_1027 is an XGBoost public notebook you can find [here](https://www.kaggle.com/code/jiaoyouzhang/calorie-only-xgboost). Thanks to @jiaoyouzhang 

Notebook #F_1023 -> CatBoost public notebook. Many thanks to @chrisk321 .You can find it  [here.](https://www.kaggle.com/code/chrisk321/lb-0-05696-ps5e5-solo-cb-model)

The rest below are all tensorflow from my private notebooks:
Notebook #F_1025, F_1021 and F_1024 -> Tensorflow

I did not bother optimizing or doing some unique stuff on the public notebooks. I just mixed this up with my Tensorflow models and call it a day. 

Logs from Hill Climbing algo:

`Current Best model:
F_1023
Models to add to the best model: 
Starting RMSE:  [0.058812503]
Iteration: 1, Model added: F_1025, Best weight: 0.41, Best RMSE: 0.05860217
Iteration: 2, Model added: F_1027, Best weight: 0.27, Best RMSE: 0.05848809
Iteration: 3, Model added: F_1021, Best weight: 0.12, Best RMSE: 0.05847134
Iteration: 4, Model added: F_1024, Best weight: 0.05, Best RMSE: 0.05846814
complete`



