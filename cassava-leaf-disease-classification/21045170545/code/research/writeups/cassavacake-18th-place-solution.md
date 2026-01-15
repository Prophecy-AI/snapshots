# 18th Place Solution

**Rank:** 18
**Author:** FGPC
**Collaborators:** FGPC
**Votes:** 17

---

I would like to give thanks to all participants who shared their wealth of knowledge and wisdom for this competition thru notebook and discussion sharing.

My submitted solution consist of simple average of 3 models namely: 5 folds gluon_seresnext101_32x4d, tf_efficientnet_b4_ns and seresnext50_32x4d. 

The following are the final scores of my model:

gluon_seresnext101_32x4d
	CV: 0.9012; Public LB: 0.9011; Private LB: 0.8993
	
tf_efficientnet_b4_ns
	CV: 0.9013; Public LB: 0.9011; Private LB: 0.8970

seresnext50_32x4d
	CV: 0.9007; Public LB: 0.8978, Private LB: 0.8987

Simple Average
	Public LB: 0.905; Private LB: 0.9013

Training Methodology
The parameters used during stages of model training:

1st stage
image size: 320x320 or 384x384
10 epochs
learning rate: 1e-4
train set: 2019 + 2020 data
validation set: 2020
scheduler: GradualWarmupScheduler
criterion: TaylorCrossEntropyLoss with label smoothing=0.2
optimizer: SAM + Adam
augmentation:  RandomResizedCrop, Transpose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, Cutout

2nd stage
image size: 512x512
25-40 epochs
learning rate: 1e-4
train set: 2020 data
validation set: 2020 data
scheduler: GradualWarmupScheduler
criterion: Combo Loss (see notebook in detail)
optimizer: SAM + AdamP
augmentation:  RandomResizedCrop, Transpose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, Cutout + Cutmix

Finetune stage
5 epochs
learning rate: 1e-5
train set: 2020 data
validation set: 2020 data
scheduler: no scheduler
criterion: Combo Loss (see notebook in detail)
optimizer: SAM + AdamP
augmentation:  hard albumentation-based augmentations

You may check this notebook for reference:
https://www.kaggle.com/projdev/base-notebook-cassava-leaf-disease-classification

I believe that the combination of SAM + AdamP optimizer provides the model robustness in terms of prediction generalization and the key to survive in competition shake-up.

A possible improvement for my model score is using SAM + SGDP and add FocalCosineLoss in Combo loss and find the right learning rate as it converges quickly. Unfortunately, I have no available GPU resources to re-train my model as SAM optimizer double the training time and I spent too much time on finding the right parameters for my model training.

Once again, thank you very much.