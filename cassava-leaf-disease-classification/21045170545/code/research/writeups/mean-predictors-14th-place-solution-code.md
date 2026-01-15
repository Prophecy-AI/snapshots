# 14th Place Solution & Code

**Rank:** 14
**Author:** Nikita Kozodoi
**Collaborators:** Nikita Kozodoi, lizzzi1
**Votes:** 86

---

## Summary

First, I would like to say thanks to my teammate @lizzzi1 for the great work and to Kaggle for organizing this competition. This was a very interesting learning experience. We invested a lot of time into building a comprehensive PyTorch GPU/TPU pipeline and used Neptune to structure our experiments. 

Our final solution is a stacking ensemble of different CNN and ViT models; see the diagram below. Because of the label noise and the evaluation metric, trusting CV was very important to survive the shakeup. Below I outline the main points of our solution in more detail. 
![cassava](https://i.postimg.cc/d1dcZ6Zv/cassava.png)

## Code
- [Kaggle notebook reproducing our submission](https://www.kaggle.com/kozodoi/14th-place-solution-stack-them-all)
- [GitHub repo with the training codes and notebooks](https://github.com/kozodoi/Kaggle_Leaf_Disease_Classification)

## Data
- validating on 2020 data using stratified 5-fold CV
- augmenting training folds with:
   - 2019 labeled data
   - up to 20% of 2019 unlabeled data (pseudo-labeled using the best models)
- removing duplicates from training folds based on hashes and DBSCAN on image embeddings
- flipping labels or removing up to 10% wrong images to handle label noise (based on OOF)

## Augmentations
```
- RandomResizedCrop
- RandomGridShuffle 
- HorizontalFlip, VerticalFlip, Transpose
- ShiftScaleRotate
- HueSaturationValue
- RandomBrightnessContrast
- CLAHE
- Cutout
- Cutmix
```
- TTA: averaging over 4 images with flips and transpose
- image size: between 384 and 600

## Training
- 10 epochs of full training + 2 epochs of fine-tuning the last layers
- gradient accumulation to use the batch size of 32
- scheduler: 1-epoch warmup + cosine annealing afterwards
- loss: Taylor or OHEM loss with label smoothing = 0.2

## Architectures
We used several architectures in our ensemble:
- `swsl_resnext50_32x4d` - x14
- `tf_efficientnet_b4_ns` - x8
- `tf_efficientnet_b5_ns`- x7
- `vit_base_patch32_384` - x1
- `deit_base_patch16_384` - x1
- `tf_efficientnet_b6_ns` - x1
- `tf_efficientnet_b8` - x1

ResNext and EfficientNet performed best, but transformers ended up being very important to improve the ensemble. Two EfficientNet models also included a custom attention module. Some models were pretrained on the PlantVillage dataset, others started from ImageNet weights.

## Ensembling

Having done many experiments allowed us to inject a lot of diversity in the ensemble. Our best single model scored **0.8995 CV.** A simple majority vote with all models got **0.9040 CV** but we wanted to go beyond that and explored stacking.

Stacking was done with a LightGBM meta-model trained on the OOFs from the same CV folds. Apart from the 33 models described above, we included 2 "special" EfficientNet models:
- pretrained ImageNet classifier predicting ImageNet classes (from 1 to 1000)
- binary classifier predicting sick/healthy cassava (0/1)

Including the last two models and optimizing the meta-classifier allowed us to reach **0.9067 CV**, which was our best. To fit all 33+2 models into the 9-hour limit, we only used weights from a single fold for each of the base models. The inference took 8 hours and scored **0.9016** on private LB just a few hours before the deadline. It was tempting to choose a different submission since stacking only got **0.9036** on public LB (outside of the medal zone), but we trusted our CV and ended up rising to the 14th place. 

Let me know if you have any questions and see you in the next competitions! 😊