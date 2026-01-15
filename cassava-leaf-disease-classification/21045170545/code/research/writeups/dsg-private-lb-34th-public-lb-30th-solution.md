# Private LB 34th / Public LB 30th solution

**Rank:** 34
**Author:** anonamename
**Collaborators:** Kon, anonamename, Shun Fukuda, dgkyoro, SiNpcw
**Votes:** 12

---

Thank you to all my teammates and participants. I would like to share a summary of our solution.

※Some teams were removed. I changed Topic Title because our LB changed (private LB 38→34th, public LB 31→30th).

# Summary

![img1](https://github.com/riron1206/kaggle_Cassava/blob/master/Cassava_Leaf_Disease_Classification_Model_Pipeline.jpg?raw=true)

Private LB 34th / Public LB 30th Inference Notebook: 
https://www.kaggle.com/anonamename/cassava-emsemble-v2-stacking?scriptVersionId=54065022

### TTA
We increased number of TTA for the 224*224 Vit model because accuracy of TTA using RandomResizedCrop with small image size was not stable.
Since public LB did not increase with stronger augmentation such as RandomBrightnessContrast, we only did flip.
Depending on random number of TTA, public LB changed by about 0.001-0.003, but this problem could not be solved.

### Blending
We blended ViT-B/16, EfficientnetB4, ResNeSt101e, and SE-Resnext50_32x4d model.
We used Unsupervised Data Augmentation (Semi-Supervised Learning) or BYOL (Self-Supervised Learning) on 2019 data, but it did not contribute much to accuracy improvement.
By averaging k-fold confidence and blending five or more models, we were able to achieve a stable public LB over 0.900.
We blended based on confusion matrix to get a higher cv, but  private LB became worse.

### Stacking
We were able to increase public LB from 0.905 to 0.907 by stacking using Conv2d+MLP.
We used predictive label of blended model as Pseudo label.
To avoid overfiting, we trained the model by adding gaussian noise to the feature.

-----------------------------------------------------------------------------------------

### Comment
For simple average blending without stacking, there was a submit that was 0.902 for both public and private LB.
![img2](https://github.com/riron1206/kaggle_Cassava/blob/master/sub_img.png?raw=true)

As discussed in other discussions, simple average blending was better for noisy test data sets.