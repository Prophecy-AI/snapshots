# 2nd place solution overview

**Rank:** 2
**Author:** hiromu
**Collaborators:** hiromu, Y. O., m.y.
**Votes:** 46

---

First of all, we'd like to thank the Kaggle team for holding this interesting competition. And, I want to thank my teammates( @yuooka, @futureboykid) for their hard work.

This post is an overview of our solution. Our solution consists of four major parts(preprocessing, base-model training, reranking-model training, postprocessing).

# Preprocessing
We simply apply the postprocessing method to preprocessing refer to [this post](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/154415#864905). Using this method, we can train models with less noise.

ex) selected_text "e fun" -&gt; "fun"

# Base-model training
Our training method is fundamentally the same as [this great kernel](https://www.kaggle.com/abhishek/roberta-inference-5-folds).  
## Model Architecture
We have tried so many model architectures... However, we finally got to use these two RoBERTa pretrained on SQuAD2.

1. Using the 11th(base) or 23th(large) hidden layer as output. ([This post](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/148417#833085))

2. Using a trainable vector and apply softmax over it &amp; multi dropout.  ([Google Quest 1st](https://github.com/oleg-yaroshevskiy/quest_qa_labeling/blob/master/step5_model3_roberta_code/model.py)) (We will refer to it hereafter as MDO.)

## Loss
We tried a lot, but finally, choose simple CrossEntropyLoss.
```
class CROSS_ENTROPY:
    def __init__(self):
        self.CELoss = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, pred, target):
        s_pre, e_pre = pred.split(1, dim=-1)
        s_tar, e_tar = target.split(1, dim=-1)

        s_loss = self.CELoss(s_pre.squeeze(-1), s_tar.squeeze(-1))
        e_loss = self.CELoss(e_pre.squeeze(-1), e_tar.squeeze(-1))

        loss = (s_loss + e_loss) / 2
        return loss
```

## Training Strategy
We suffered from learning instability. There are two main ideas to get a stable result.
### SentimentSampler
Most of the public kernel using stratified-KFold by sentiment. But, we decided that wasn't enough. Seeing [this post](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/138520), there is a huge difference among the sentiment.

So, we adopt SentimentSampler to equalize the imbalance within the batch.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1497263%2F4a7afe89246c305a5d8e1f648ead65ef%2F2020-06-17%2010.32.33.png?generation=1592357573339723&amp;alt=media)
### SWA
We found the instability of the validation score while training (~±0.001 varies with just 1 iteration!)

So, we adopt [SWA](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/) to stable the result.

This makes it possible to get stable results with 10~50 iteration monitoring, so that we could get results not overfitted to validation set (additionally validation time efficient.)

## Ensemble
Using two different seeds (seed averaging)

RoBERTa Base 11th layer + RoBERTa Large 23th layer + RoBERTa Base MDO + RoBERTa Large MDO

4 models * 2 seeds = Total 8 models  

# Reranking-model training (public +0.002, private +0.003~5)
## Approach
We believe this part is the unique and the most different from other teams. This approach is based on the following idea. "Creating multi candidates and choose best one."

Step1 Calculating top n score based on the start &amp; end value(apply softmax) from base-model.

Step2 Creating candidates based on the Step1 score. Candidates include selected_text &amp; Jaccard_score &amp; Step1_score.

Step3 Training RoBERTa Base model. (Target is candidate's jaccard_score)

Step4 Sorting candidates by Step3 predicted value + Step1 score * 0.5. And choose the best one as the final answer.

## SequenceBucketing
We have built over 50 models until above (base-models &amp; reranking models). In order to finish inference within the limited time, we have decided to chose SequenceBucketing even some models had not been trained with it.

 In this case, each batch contains the same text and little different candidate. 

Therefore, inference time speed up x2 and surprisingly got a better result than not using. We need to find out why...

# Postprocessing (public +0.01, private +0.012)
Just like any other team found magic, we focused on the extra space.
```
def pp(filtered_output, real_tweet):
    filtered_output = ' '.join(filtered_output.split())
    if len(real_tweet.split()) &lt; 2:
        filtered_output = real_tweet
    else:
        if len(filtered_output.split()) == 1:
            if filtered_output.endswith(".."):
                if real_tweet.startswith(" "):
                    st = real_tweet.find(filtered_output)
                    fl = real_tweet.find("  ")
                    if fl != -1 and fl &lt; st:
                        filtered_output = re.sub(r'(\.)\1{2,}', '', filtered_output)
                    else:
                        filtered_output = re.sub(r'(\.)\1{2,}', '.', filtered_output)
                else:
                    st = real_tweet.find(filtered_output)
                    fl = real_tweet.find("  ")
                    if fl != -1 and fl &lt; st:
                        filtered_output = re.sub(r'(\.)\1{2,}', '.', filtered_output)
                    else:
                        filtered_output = re.sub(r'(\.)\1{2,}', '..', filtered_output)
                return filtered_output
            if filtered_output.endswith('!!'):
                if real_tweet.startswith(" "):
                    st = real_tweet.find(filtered_output)
                    fl = real_tweet.find("  ")
                    if fl != -1 and fl &lt; st:
                        filtered_output = re.sub(r'(\!)\1{2,}', '', filtered_output)
                    else:
                        filtered_output = re.sub(r'(\!)\1{2,}', '!', filtered_output)
                else:
                    st = real_tweet.find(filtered_output)
                    fl = real_tweet.find("  ")
                    if fl != -1 and fl &lt; st:
                        filtered_output = re.sub(r'(\!)\1{2,}', '!', filtered_output)
                    else:
                        filtered_output = re.sub(r'(\!)\1{2,}', '!!', filtered_output)
                return filtered_output

        if real_tweet.startswith(" "):
            filtered_output = filtered_output.strip()
            text_annotetor = ' '.join(real_tweet.split())
            start = text_annotetor.find(filtered_output)
            end = start + len(filtered_output)
            start -= 0
            end += 2
            flag = real_tweet.find("  ")
            if flag &lt; start:
                filtered_output = real_tweet[start:end]

        if "  " in real_tweet and not real_tweet.startswith(" "):
            filtered_output = filtered_output.strip()
            text_annotetor = re.sub(" {2,}", " ", real_tweet)
            start = text_annotetor.find(filtered_output)
            end = start + len(filtered_output)
            start -= 0
            end += 2
            flag = real_tweet.find("  ")
            if flag &lt; start:
                filtered_output = real_tweet[start:end]
    return filtered_output
```
### This post is a brief version of our solution.  
### Please looking forward to the detail explanation from my teammates :)
Reranking-model https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159315
Postprocessing https://www.kaggle.com/futureboykid/2nd-place-post-processing