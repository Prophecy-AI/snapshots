# Private LB 36th place/Public LB 1138th place [Silver medal] solution

**Rank:** 36
**Author:** ilovescience
**Collaborators:** ilovescience, Zach Mueller, Abhishek Thakur, Mr_KnowNothing, Aditya Soni
**Votes:** 51

---

Wow what a competition! We jumped from our public leaderboard position of 1138th place to the private position of 36th place, giving us a silver medal. Our solution is quite simple, we didn't utilize any techniques to deal with noise. Did I mention our submissions finished just minutes before the deadline?

# Solution:

The components of our silver medal solution:

1. EfficientNet-B7
2. EfficientNet-B3a from timm library
3. SE-ResNext50

**EfficientNet-B7:**
This is basically the same as [this kernel](https://www.kaggle.com/abhishek/tez-faster-and-easier-training-for-leaf-detection) but with an EfficientNet-B7 model, 20 epochs, and image size of 512x512. 5-fold CV of:
This model was submitted with a 3x TTA of the following augmentations:
```
    RandomResizedCrop, 
    Transpose(p=0.5), 
    HorizontalFlip(p=0.5), 
    VerticalFlip(p=0.5), 
    RandomBrightnessContrast(
        brightness_limit=(-0.1,0.1), 
        contrast_limit=(-0.1, 0.1), 
        p=0.5
    ),
```
We will note that during the private submissions leak (~1.5 months ago), we noticed this model leaderboard submission was 9th place. 
	
**EfficientNet-B3a:**

Training for this model was inspired by the training procedure in [this kernel](https://www.kaggle.com/keremt/progressive-label-correction-paper-implementation) (but without label correction). Basically, the EfficientNet-B3a (a version of EfficientNet-B3 with weights trained by Ross Wightman) with the model body frozen and the fastai model head unfrozen was 
trained for 3 epochs. Following this, the whole model was unfrozen and trained for 10 epochs. 512x512 images were used, batch size of 32. The augmentations used were as follows:
```
   Dihedral(p=0.5),
   Rotate(p=0.5, max_deg=45),
   RandomErasing(p=0.5, sl=0.05, sh=0.05, min_aspect=1., max_count=15),
   Brightness(p=0.5, max_lighting=0.3, batch=False),
   Hue(p=0.5, max_hue=0.1, batch=False),
   Saturation(p=0.5, max_lighting=0.1, batch=False),
   RandomResizedCropGPU(size, min_scale=0.4),
   CutMix()
```
Label smoothing loss function with Ranger optimizer was used. The CutMix+label smoothing was quite helpful, and we noticed a significant CV boost because of this. The model with the best validation loss was saved. This resulted in a model with a 5-fold CV of: 0.8914
The model was submitted with 5x TTA with the training augmentations (`learn.tta(n=5,beta=0)`).

**SE-ResNext50:**

I just changed the model from the previous pipeline to SE-ResNext50. A 5-fold CV of: 0.8938

These three models were ensembled and submitted. 

Public LB: 0.898
Private LB: 0.900

We also had another SE-ResNext50 model with different augmentations, LR schedule, etc. along with stochastic weight averaging during the end of training. This was submitted with the EfficientNet-B3a and the EfficientNet-B7 as an ensemble, and this got Public LB: 0.897, Private LB: 0.901 (potential gold territory), but this was not selected unfortunately. However, we are grateful that we even managed to make a 1135 rank jump!

**A few of the other things we tried:**
	- Tried Bi-tempered logistic loss, observed no CV improvement
	- Tried progressive label correction, observed no CV improvement
	- Tried pseudolabeling on the test set, it was a last minute idea and public LB was 0.139 so likely an error in implementation

**Reflections:**
	- This competition was quite interesting, even though the dataset was quite messy and filled with errors
	- Even with many errors, the model training was robust, leading to decent performance on the CV and private leaderboard
        - While we all started out quite enthusiastic regarding this competition, other tasks/duties took higher priority for many of us, and only a couple of us dedicated significant time to it. I wonder what the situation would be if all team members could dedicate more time to the competition. Would we reach an even higher place? Or would we overfit to the leaderboard? 

Thanks to my great teammates, especially @tanulsingh077 and @abhishek!