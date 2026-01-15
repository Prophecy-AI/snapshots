# 19th Place Solution 

**Rank:** 19
**Author:** Sam
**Collaborators:** Sam
**Votes:** 7

---

**Notebook :** [link](https://www.kaggle.com/code/samlakhmani/rank-19-91-076-oof-strategy)

**I kept the original data as part of the training set only**
The idea was the in the CV score, I wanted the original data to not influence the accuracy of the CV. Thus I appended the original data. You can observe the model function in the above code. 
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3298303%2Fc6f2c44055fc7cdc8850ed52a2ed479e%2FScreenshot%202024-03-06%20at%2010.50.26%20AM.png?generation=1709702445730912&alt=media)

**Choosing Folds** 
I experimented with optimization for 10CV 
5CV performed better than 10CV and thus I choose 5CV 
```python
Looks like this remains the only difference between the 19th place and 2nd palace :P 
2nd place used 20CV
```

**Choosing no of time to append Original data to training**
If you look at these two notebooks you will be able to observe experiments I had done with the multiplier. Even My experiments gave 4 as the best multiplier for LGBM, and 1 as a multiplier to the XGB model. 

![Multiplier code](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3298303%2F134334a34b90438c53c239d19413839d%2FScreenshot%202024-03-06%20at%2010.15.10%20AM.png?generation=1709700333809544&alt=media)

![Image of accuracy vs no of time data set is appended](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3298303%2F1d79d398dfece58fc64f667c28fd4abc%2FScreenshot%202024-03-06%20at%2010.11.00%20AM.png?generation=1709700240794258&alt=media)

**Thresholding Optimization** 
[This](https://www.kaggle.com/code/samlakhmani/easy-92-196-single-model?scriptVersionId=163207315) is the notebook where I introduced threshold optimization to this competition, post which a lot of people started implementing it. My intention of making it public was to learn on how the implementation can be improved. **It would me a lot to know the impact of the model without the optimization. Please share.** 
Requesting to Upvote the Notebook ^ 