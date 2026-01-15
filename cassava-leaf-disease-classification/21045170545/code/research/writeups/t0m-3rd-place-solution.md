# 3rd Place Solution

**Rank:** 3
**Author:** T0m
**Collaborators:** T0m
**Votes:** 164

---

## Acknowledgements
Thanks to Kaggle and hosts for holding this competition.
This is my first gold medal and I finally became Competition Master :)

# Summary
[update 2021.03.02]
I made a mistake, vit_base_patch16_384's final layer is not multi-drop but simple linear.
I found a bug during the verification process.

My final submission is ensemble of three ViT models; summarized below.
[![summary.png](https://i.postimg.cc/HsjCzjjw/summary.png)](https://postimg.cc/sQddZFXM)
# Model
### vit_base_patch16_384
  - img_size = 384 x 384
  - 5x TTA 
  - Public  : 0.9059
  - Private : 0.9028

This is my best single model

### vit_base_patch16_224 - A
  - img_size = 448 x 448
  - 5x TTA 
  - weight calculation pattern A
  - Public  : 0.9030
  - Private : 0.8990

To adapt vit_base_patch_224(expected image size is 224 x 224) for 448 x 448 image,
after augmentation, divide it into four parts and input each image into the model. And then, they are adapted the weighted average using calculated weights at attention layer, and finally output prediction using Multi-Dropout Linear.
　
### vit_base_patch16_224 - B
  - img_size = 448 x 448
  - 5x TTA 
  - weight calculation pattern B
  - label smoothing, alpha=0.01
  - Public  : 0.9034
  - Private : 0.8952
　
### Weighted Averaging
  - Public  : 0.9075
  - Private : 0.9028

Tried a bunch of pretrained models but ViT model works the best at Public LB. 
The bigger the image size, the better cv score, but I thought it is overfitting. So I dropped efficient-net and se-resnext, etc with large image size in the early stages.

## Some Settings
- 5fold StratifiedKFold
- Using 2020 & 2019 data

### Augmentation
I tried some types of augmentations, but finally adopted simple one.
The reason why is the same of chose not large size image, overfitting.

```python
if aug_ver == "base":
    return Compose([
        RandomResizedCrop(img_size, img_size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
])
```
### LR Scheduler
- LambdaLR
```python
self.scheduler = LambdaLR(
    self.optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + epoch)
)
```
![lr.png](https://i.postimg.cc/Twv8mmCW/2021-02-21-9-29-45.png)

### Scores

![scores.png](https://i.postimg.cc/hvhFzmdT/2021-02-21-23-00-39.png)

training code
https://github.com/TomYanabe/Cassava-Leaf-Disease-Classification

Thank you for reading :)