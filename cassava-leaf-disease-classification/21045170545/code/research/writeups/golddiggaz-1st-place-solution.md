# 1st Place Solution

**Author:** Jannis
**Rank:** 1
**Votes:** 294

---

Our overall strategy was to test as many models as possible and spend less time on fine-tuning. The goal was to have many diverse models for ensembling rather than some highly tuned ones.

In the end, we had tried a variety of different architectures (e.g., all EfficientNet architectures, Resnet, ResNext, Xception, ViT, DeiT, Inception and MobileNet) while working with different pre-trained weights (trained e.g. on Imagenet, NoisyStudent, Plantvillage, iNaturalist...) some of which were available on Tensorflow Hub. 

<b>Our winning submission was an ensemble of four different models.</b>
![Final Model](https://i.ibb.co/fCdNjTY/1stplace.png)

The final score on the public leaderboard was <b>91.36%</b> and <b>91.32%</b> on the private leaderboard. We opted to turn in this combination as it achieved a higher CV score than other combinations (which sometimes scored slightly better on the public leaderboard). We tested some of the models separately on the leaderboard (public/private): <b>B4: 89.4%/89.5% , MobileNet: 89.5%/89.4%, ViT:~89.0%/88.8%</b> 

The others were only evaluated using cross-validation.

Overall, we can conclude that the key to victory was the use of CropNet from Tensorflow Hub, as it brought a lot of diversity to our ensemble. Although it did not perform better on the leaderboard as a standalone model than the other models, the ensembles that used this model brought a significant boost on the leaderboard.

At this point I would like to thank hengck23, whose [discussion post](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/199276) brought this model to our attention (as well as other pre-trained models for image classification available on Tensorflow Hub).

Furthermore, we would like to thank all the participants who have helped us learn a lot in this contest through their contributions here in the board and their published notebooks. Finally, we would also like to thank the Competition Host, who made this exciting competition possible by releasing the data.

You can find our inference code in this notebook: <a href="https://www.kaggle.com/jannish/final-version-inference">Inference Notebook</a>

And you can find the according training code in these notebooks:
<ul>
<li><a href="https://www.kaggle.com/hiarsl/cassava-leaf-disease-resnext50">ResNext50_32x4d (GPU Training)</a></li>
<li><a href="https://www.kaggle.com/sebastiangnther/cassava-leaf-disease-vit-tpu-training">ViT (TPU Training)</a></li>
<li><a href="https://www.kaggle.com/jannish/cassava-leaf-disease-efficientnetb4-tpu">EfficientNet B4 (TPU Training)</a></li>
</ul>

Detailed information about the configuration & fine-tuning of our used models:

We used a <b>ResNeXt </b>model of the structure “resnext50_32x4d” with the following configurations:

<ul>
<li>image size of (512,512)</li>
<li>CrossEntropyLoss with default parameters</li>
<li>Learning rate of 1e-4 with “ReduceLROnPlateau” scheduler based on average validation loss (mode=’min’, factor=0.2, patience=5, eps=1e-6)</li>
<li>Train augmentations (from the Albumentations Python library): RandomResizedCrop, Transpose, HorizontalFlip. VerticalFlip, ShiftScaleRotate, Normalize</li>
<li>Validation augmentations (from the Albumentations Python library): Resize, Normalize</li>
<li>5-fold-CV with 15 epochs (after the 15 training epochs, we always chose the model with the best validation accuracy) (same data partitioning as for the other trained models)</li>
<li>For inference we used the same augmentations as for validation (i.e., Resize, Normalize)</li>
</ul>

We used the <b>Vision Transformer Architecture </b> with ImageNet weights (ViT-B/16)

<ul>
<li>Custom top with Linear layer; Image size of (384,384)</li>
<li>Bit Tempered Logistic Loss (t1 = 0.8, t2 = 1.4) and label smoothing factor of 0.06</li>
<li>We chose a learning rate with a Cosine annealing warm restarts scheduler (LR =  1e-4 / 7 [7: Warm up factor], T0= 10, Tmult= 1, eta_min=1e-4, last_epoch=-1). A batch accumulation for backprop with effectively larger batch size</li>
<li>Train Augmentations (RandomResizedCrop, Transpose, Horizontal and vertical flip, ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast, Normalization, CoarseDropout, Cutout)</li>
<li>Validation Augmentations (Horizontal and vertical flop, CenterCrop, Resize, Normalization)</li>
<li>5-fold-CV with 10 epochs, we took for each fold the best model (based on the validation accuracy) </li>
<li>For inference we used the following augmentations: CenterCrop, Resize, Normalization</li>
</ul>

We tried different EfficientNet architectures but finally only used a <b>B4 with NoisyStudent weights</b>:
<ul>
<li>Drop connect rate 0.4, custom top with global average pooling and dropout layer (0.5)</li>
<li>Sigmoid Focal Loss with Label Smoothing 
(Gamma=2.0, alpha=0.25 and label smoothing factor 0.1)</li>
<li>Learning rate with warmup and cosine decay scheduler (ranging from 1e-6 to a maximum of 0.0002 and back to 3.17e-6)</li>
<li>Augmentations (Flip, Transpose, Rotate, Saturation, Contrast and Brightness and some random cropping)</li>
<li>Adapting the normalization layer with the global mean and deviation of the 2020 Cassava dataset</li>
<li>5-fold-CV with 20 epochs with early stopping and callback for restoring weights of best epoch</li>
<li>Final model was trained for 14 epochs on whole competition data set</li>
<li>For inference we used simple test time augmentations (Flip, Rotate, Transpose). To do so, we cropped 4 overlapping patches of size 512x512px from the .jpg images (800x600px) and applied 2 augmentations to each patch. We retained two additional center-cropped patches of the image to which no augmentations were applied. To get an overall prediction, we took the average of all these image tiles. </li>
</ul>

Finally, our ensemble included a pretrained <b>CropNet (MobileNetv3)</b> Model from Tensorflow Hub:
<ul>
<li>We used a pretrained model from TensorFlow Hub called <a href="https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2">CropNet </a> which was specifically
trained to detect Cassava leaf diseases </li>
<li>The CropNet model is based on the MobileNetV3 architecture. We decided not to do any 
additional fine-tuning of that model.</li>
<li>As stated in the description the images must be rescaled to 224x224 pixel which is pretty
small. We achieved good results by not just resizing our 512x512 training images but to center
crop them first.</li>
<li>As the notebooks had to be submitted without internet access, it was necessary to cache the
model before including it. You can find more information on this on the official <a href="https://www.tensorflow.org/hub/caching">TF Hub website</a> or alternatively in this post on <a href="https://xianbao-qian.medium.com/how-to-run-tf-hub-locally-without-internet-connection-4506b850a915">Medium</a>. </li>
</ul>

For the ensembling, we experimented with different methods and found that in our case a stacked-mean approach worked best. For this purpose, the class probabilities returned by the models were averaged on several levels and finally the class with the highest probability was returned. 

<b>Our final submission first averaged the probabilities of the predicted classes of ViT and ResNext. This averaged probability vector was then merged with the predicted probabilities of EfficientnetB4 and CropNet in the second stage. For this purpose, the values were simply summed up.
</b>

Another solution which also generated good results on the leaderboard was finding weights before calculating the mean of models using an optimization. You can find the code for that in our published notebooks. Generally, we were surprised how stable the solutions with optimized weights were. It turned out that they only had small differences (often +/-0.1%) between our CV scores and the leaderboard score.

One thing which didn’t work out in our use case was an ensemble approach with an additional classifier (Gradient Boosted Trees) stack on top of our models. We did several experiments using also additional features, like e.g., the entropy of the model’s prediction, however we were not able to build a solution which generalized good enough. 


Thanks for reading. 




