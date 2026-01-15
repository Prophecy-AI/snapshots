# Quick 1st place solution overview before the night 

**Author:** Theo Viel
**Rank:** 1
**Votes:** 123

---

### Update : 
- Training notebook for our 2nd level models : https://www.kaggle.com/theoviel/character-level-model-magic/
- More detailed write-up : https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159477

This is a short thread to rapidly present a solution, we will work on a more detailed one tomorrow. 

Huge thanks to my teammates @cl2ev1, @aruchomu and @wochidadonggua for the great work, we definitely wouldn't have gone this far if we were not together. 

Our whole solution can be illustrated in the following pipeline.


![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2062758%2F9848f1052e0108f6257fa56f1233a9d1%2Fpipe.png?generation=1592353191974814&amp;alt=media)

The idea is to use transformers to extract token level start and end probabilities. Using the offsets, we can retrieved the processed probabilities for the input text.

We then feed these probabilities to a character level model. 

The tricky part is to concatenate everything correctly, such as explained [here](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159254) 

And then... TADAM ! 
No post-processing. Just modeling. 

We selected two models that scored public LB 0.734  / CV 0.736+. They use 4 different character level model each, with a big variety of transformers. Final private scores are 0.735 and 0.736 :)

Thanks for reading ! 