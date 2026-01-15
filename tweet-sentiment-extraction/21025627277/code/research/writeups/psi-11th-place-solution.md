# 11th place solution

**Rank:** 11
**Author:** Psi
**Collaborators:** Psi
**Votes:** 67

---

Thanks a lot to Kaggle for hosting this competition! It was quite a fun, but very exhausting ride and I am very happy with achieving my goal of solo gold!

### Data 

As we all have seen, there is quite some noise in the data. Many people found pre- or post-processing techniques for dealing with it. Others had really quite clever model adaptions to address it. I personally think there are actually two types of facets in the data. Let us discuss the following example:

`That's awesome!!!` --&gt; `s awesome!`

1. Let us first focus on the second selected word `awesome!`. Here, we can see stark differences in how the end (special) characters of words are selected. I believe this is due to human annotators having different preferences, and is real noise in the data. I personally would not directly know if I select all exclamation marks, only one, or none. So that there is difference in the data seems natural to me.

2. Unfortunately, there appears to also be a form of leak in the data (see first word). When the data was first fixed, we learned that the labeling service only provided the indices back, but did some preprocessing on their own. We learned that the escaped HTML tags and that caused a shift in the indices. Unfortunately, this was not the only thing they did, they also apparently replaced multiple spaces with single ones, which again lead to wrong indices being used by Kaggle.

I personally think the first case is interesting to model, and the second case is based on wrong data processing upfront. I was identifying the second issue quite early, but just could not find a reliable way to find a PP that works. I don't think it is a consistent error, as it is not always the case, as for example imminent from neutral data where you have a nice baseline with selecting the full text. Many times if there is a space in the beginning, the selected text is still the full text. So maybe there are also two or more labeling processes in play, some are correct, some not. But I just could not figure it fully out.

### Tokenization

I decided to focus on tokenization to address some of these issues. With default Roberta tokenization,  you can always only predict `!!!` as those tokens are combined into a single one. This heavily limits your model to learn the noise and differences here, and if majority of people select a single exclamation mark, you won't be able to predict it, except you do some PP. What I did though is to modify the `merges.txt` file of the tokenization. This one has for example a line `!! !` which means that those subsequent character spans are always merged to a single token. I started removing all those lines with dots and exclamation marks, and got good boosts on CV quite away. In the end, I removed all special character merges:

```
f = open("merges12.txt", 'w')
for line in open("merges.txt"):
    cols = line.strip().split()
    if cols[0] == "Ġ":
        if cols[1].isalpha() or len(cols[1]) == 1:
            f.write(line)
    elif cols[0].isalpha() and cols[1].isalpha():
        f.write(line)
```

I wrote more about it in a separate thread: https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159364 and feel free to try it out: https://www.kaggle.com/philippsinger/roberta-base-squad2

By doing so, I tried to increase the jaccard score I can get on the original selected text with given tokenization, which is out of the box not perfect. You can remove as many merges as you want, and can get 1.0 jaccard score, by e.g., further removing merges like `Ġg onna`, but in the end this didn't boost my scores, so I sticked to just the special characters.

I believe my approach is very good at dealing with the real noise in the data (point 1 from above), but not so good with dealing with the artificial noise (point 2 from above). What helped with the artificial one is to keep all spaces intact, so the model learns it a bit. But top solutions handled this way better with either extra heads or post processing.

### Models

I tried quite a bunch of models, but as for most of us Roberta base worked fine enough. I got a tiny boost by using a version pre-trained on Squad2. My head is pretty standard, with the exception that I added a trick I read in [TF competition](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127545): I concatenate the output of the stark token representations to the input of the end token representations, so the end layer knows what start token was predicted.

```
start_logits = self.l1(o)
start_token = torch.gather(o,1,start_logits.argmax(dim=1, keepdim=True).repeat(1,1,o.size(2)))
o2 = torch.cat([o,start_token.repeat(1,o.size(1),1)], dim=2)
end_logits = self.l2(o2)
```

I fit models with Radam or AdamW over 5-8 epochs. In epoch 1 I keep learning rate constant, and after that I do cosine decay. My starting transformer LR is 1e-4.

A "trick" I found is to only fit on all data on first epoch, and then drop neutral data and only train on positive and negative. For neutral data I fit separate models early stopping on neutral eval only. In general, I do early stopping, but sometimes also fixed epochs, and different types of cvs.

As I always do, I never looked at single model OOF scores to make judgements, but always fitted each fold at least 5 times and blended the result. This is way better indicative of test submissions you do (where you at least blend all folds) and also gave me a quite good understanding of the random range. 

### Pseudo tagging

I had high hopes in the beginning for pseudo tagging in this competition, but it was way less helpful than I thought it would be. At first I was hopeful when adding public test to training boosted me from 724 to 728 on the first attempt, but it was lucky. The models are in general really, really good in memorizing here, so for example if you add 728 to train, and then predict again on test, you will always be in the 728 range.

However, based on that I found out that you can use it well to compress models. So for example, you can run 5-bag CV, blend them, predict and then feed this together with train data to another model, and the model is good at predicting unseen data with fewer fits.

### Final subs

My final subs are heavy blends on many, many model fits. To bring diversity to my subs, one of them is using a blend of regular models, and another one on only pseudo models. Both use the same models for neutral data. The first one is using also an adaption of [JEM](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/158613) that was posted. All diversity efforts were quite useless in the end, as both subs scored basically the same and both are under the top 4 subs of my potential private LBs.

### Things that didnt work

There are too many things, and I am tired to even think of them, but to mention a few:

- changing the loss function
- replacing **** with original swear words
- other models
- pretraining on twitter data or finetuning language models
- including the extra sentiment data
- any of my post-processing attempts
