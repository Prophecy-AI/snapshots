# 21st place solution

**Rank:** 21
**Author:** Dracarys
**Collaborators:** Dracarys, Manoj, Shahules, Priyanshu Kumar, ryoya
**Votes:** 40

---

Hello everyone, first of all i would like to thank my great teammates @ryoya0902, @mks2192, @shahules and @kpriyanshu256  for such an interesting competition journey. And congratulations to all the winners!

It's been a great competition, and we have spend a lot of time in this competition and finally glad to share that all the hard work paid off.

## Quick Summary of  things that worked
* RoBERTa base as our base model with MAX_LEN = 168
* Used preprocessing (this boost our local CV from 0.711 to 0.717)
* Trained for 8 folds  (8 fold was giving better CV in our case)
* At last we used the so called magic (Post Processing) (boost our LB from 0.717 to 0.724)
* Also validating multiple times per epoch has also improved our CV.


## Things that didn't worked
* We tried to solve this problem as NER, unsupervised text selection, they didn't worked.
* We tried various model architectures, but none of them worked well except `roberta-base`.
* We tried training seperate model for each sentiment but that didn't worked as well.
* We also tried a lot of preprocessing techniques like removing all the noisy samples, cleaning text, etc but none worked.
* We also tried BERTweet, it also didn't work for us.
*  We also tried augmentations like Synonym replacement, etc.
* Augmentation kernel can be found [here](https://www.kaggle.com/rohitsingh9990/data-augmentation-by-synonym-replacement)


## Preprocessing that worked
We know that there is huge amount of noise in samples where some extra space is present in between words, for example:

```
text: is back     home gonna miss every one
selected_text: onna
after_preprocessing: miss

```
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3982638%2F374c9e6d4fa5521778f8268b26bb4f53%2FScreenshot%202020-06-17%20at%207.46.08%20AM.png?generation=1592360225973868&amp;alt=media)

* The catch was you need to shift start and end indices by the number of extra spaces.
* preprocessing kernel can be found [here](https://www.kaggle.com/rohitsingh9990/preprocessing-that-worked?scriptVersionId=36556534)

## Training 

* After applying preprocessing, we trained our model for 8 folds
* We used the same model architecture as shared by @abhishek 
&gt; Note:  Need to clean the training kernel, will share soon.

## Inference

* after getting the `selected_text` from model we apply our `post processing`, it's nothing but just the reverse engineering of our preprocessing kernel shared above.
* it's almost similar to what's been discussed [here](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159245)
* And finally the inference kernel can be found [here](https://www.kaggle.com/rohitsingh9990/roberta-prepost-0-726)

At last i would like to thank @abhishek and @cdeotte for their amazing starter kernels and their spirit of public sharing, which helped other kagglers a lot.