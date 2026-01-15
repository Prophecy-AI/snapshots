# 5th solution

**Rank:** 5
**Author:** AIFIRST
**Collaborators:** Rainbow Cat, xia, AIFIRST, ggsddu
**Votes:** 45

---


First of all I want to thank my teammates here.I will briefly introduce our solution here.

## models
1.Model structure。we  design different models structure. We mainly refer to the solution of ccf internet sentiment analysis，concat different  cls embedding . here it is the link [BDCI2019-SENTIMENT-CLASSIFICATION](https://github.com/cxy229/BDCI2019-SENTIMENT-CLASSIFICATION)  
 2.We found 30 labels through analysis, one is the question-related evaluation, and the other is the answer-related evaluation. In order to make the model learn better, we have designed the q model to remove the problem-related label and the a model to process the answer Related labels。 it is better than qa models.  
3.different model  test.  roberta base &gt;roberta large &gt;xlnet base &gt;bert base &gt; t5 base.   
 
## Post-processing  
1. Analysis and evaluation methods and competition data，we use 0,1 reseting  way. it improve lb 0.05 or more.  


## Features
1.we want that our model learns features that are not only considered in text, so we add host and 
category embeeding features  annd other Statistical Features。 it improve both cv and lb about 0.005.  

##Text  clean  
1.We also did text cleaning to remove stop words and some symbols， it improve  about 0.002 



## Stacking 

1.Our Best  Private model  scored  0.42787  ,but we dont't select  it.  it is stacking  by  roberta large and roberta base and xlnet base.  
`blend.loc[:,targets] =  roberta_large_oof_test.loc[:,targets].values*0.4+0.3*roberta_base_oof_test.loc[:,targets].values+\
                        xlnet_base_oof_test.loc[:,targets].values*0.3`  

stacking improve  both cv and lb  about  0.02 .  it help much.
