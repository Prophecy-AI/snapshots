# 10th place solution

**Rank:** 10
**Author:** Yauhen Babakhin
**Collaborators:** Guanshuo Xu, Yauhen Babakhin
**Votes:** 47

---

First of all, thanks to my great teammate @wowfattie for another interesting competition journey. And congratulations to all the winners!

In this post, I will briefly describe our path in this competition and then go straight to our final model description.

## Motivation to join
I've joined this competition relatively late (about a month before the deadline). Probably, it was even an advantage, keeping in mind all the data problems in this competition.

My motivation to join this competition was to make myself familiar with the recent NLP models and approaches. The last time I was working with the text data was pre-Transformers era.
**Spoiler: I haven't learnt a lot about recent NLP trends**

I had read the basic theory of Tranformers and Bert-like models. Then it was very helpful to go through the code and Youtube videos shared by @abhishek. Thanks again! Basically I've built all my models based on this code.

## Data property (a.k.a. Magic)
After some tweaking of the initial Roberta model, I've managed to get 0.714 on the Public Leaderboard. Then I've tried to delete the preprocessing that was doing `" ".join(text.split())` and unexpectedly Public LB score jumped to 0.716.

It was at a similar time moment when the magic post has been shared. So, I decided to dig into what is hidden in the space distribution. And it occurred that the majority of the labels noise could be explained by the extra spaces in the text.

First, thanks to @maxjon for creating a [correspondence dataset](https://www.kaggle.com/maxjon/complete-tweet-sentiment-extraction-data) between the original and Kaggle datasets. I'm still not sure about the exact rules this data property follows, but here are our most frequent findings:
- Extra spaces could come either from the initial tweet text or from the deleted @mentions from the original dataset
- Start of the actual label is shifted left by the number of extra spaces
- End of the actual label is also shifted left by the number of extra spaces, but then shifted right by 2 characters

Seems like the initial labeling has been done with `" ".join(text.split())` preprocessing, and that caused the labels shift in the actual texts. Maybe someone else could shed more light on this. For example, (spaces are substituted with |):

```
# Original dataset text:
"@blueskiesxj||i|like|yours|too|||i|enjoy|your|photography.|=]"

# Kaggle dataset text:
"||i|like|yours|too|||i|enjoy|your|photography.|=]"
```
So, we could observe 2 extra spaces in the beginning of the tweet, and 2 extra spaces in the middle, 4 in total. Probably, the actual selected text was `"enjoy"`, while in the dataset it is a shifted version: `"i enj"`.

## Final model
After this finding problem is transformed from modeling into properly utilizing this data property. Firstly, we've built postprocessing that gave 0.721 Public LB for a single 5-fold Roberta.

Then, we changed a problem a little bit. Firstly, try to inverse this labeling transformation on the train data to get cleaner labels. Then fit the model on clean data, and transform predictions back to the noisy labels with a postprocessing. Such an approach allowed us to get 0.723-0.724 Public LB scores with a 5-fold Roberta model. And the validation score was closely following the Leaderboard.

Other parts of our solution include:
- Using weighted sum of cross-entropy losses (start and end) and lovasz loss for a span prediction
- Use pseudo-labels from the Sentiment140 dataset
- Ensemble multiple models. Our best submission includes 18 5-fold models: 8 Robertas; 4 Electras; 6 Berts Large

It allowed to get 0.731 Public LB and 0.728 Private LB.

To conclude, it was a nice experience cracking the problem, while I haven't tried a lot of new NLP stuff.

Also, I want to thank the [hostkey](https://www.hostkey.com/) provider for giving me a [grant](http://landing.hostkey.com/grant_for_winners) on using the machine with 4x1080TIs for this competition. It was a smooth experience, highly recommend it!