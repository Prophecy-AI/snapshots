# 7th place solution

**Rank:** 7
**Author:** Xuan Cao
**Collaborators:** Morphy, Xuan Cao, Yuanhao
**Votes:** 72

---

First of all, we want to thank kaggle for hosting the competition. Thanks for my teamates @murphy89 @wuyhbb for their hard work. Thank @abhishek  for providing a very solid baseline. This is my first NLP gold medal and I am extremely happy! 

# TLDR
Use model to predict the **ground truth** start/end indices, use post processing to capture **noise**. 

# Models
Our models are RoBERTa-base with customized headers. We have two model structures:
1. Model-1
- Concat([last 2 hidden_layers from BERT]) -&gt; Conv1D -&gt; Linear
- End position depends on start (taken from [here](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127545)), which looks like,
```python
# x_head, x_tail are results after Conv1D
logit_start = linear(x_start)
logit_end = linear(torch.cat[x_start, x_end], dim=1) 
```
2. Model-2
- Concat([last 3 hidden_layers from BERT]) -&gt; Conv1D -&gt; Linear
- Auxiliary tasks for whether the prediction is whole text (classification) and whether the token is in selected text (segmentation). 

# Preprocessing &amp; Tokenization
We used two methods to preprocess the text and selected_text:
* Method-1

1) Clean text &amp; selected_text: ```" ".join(text.split())```. 

2) Split punctuations if sentiment is not "neutral": '...' ==&gt; '. . .' (50% if sentiment != 'neutral'). 

3) Correct wrong offsets for "ï¿½" (this is not needed if you are using tokenizers 0.7.0). 

4) A token is considered as a target if ```len(selected_text) &gt; 4``` and ```char_target_pct of the token &gt;= 0.5```. 

5) Max_len = 192.

* Method-2

1) Clean label based on postprocessing method (will discuss later). 

2) Use raw text &amp; selected_text, tokenize at word level, use "Ġ" for all spaces. 

3) Dynamics padding. 

We used two patterns to build the training samples for both methods:  [sentiment][text] and 
 [sentiment][raw sentiment][text], where raw sentiment comes from the original full dataset. We didn't convert text to lower case during preprocessing, instead, we use it as an augmentation. Method-1 is used for Model-1, while Method-2 is used for Model-2. 

# Training related
* Batch size: 32
* Optimizer AdamW, weight decay 0.01
* Scheduler: cosine scheduler with warmup
* Loss: CrossEntropy with label smoothing

We finetuned Model-1 (head only) for another 3 epoches using linear scheduler with warmup (lr=1e-5). Model-2 is trained with Fast Gradient Method (FGM). Our best single model on public LB: [sentiment][raw sentiment][text] + Method-1 + Model-1, trained on a 10-fold setup (CV 0.7177, LB 0.719).

# Postprocessing
There are two parts of postprocessing: set "neutral" predictions as texts (all CV scores are after this treatment) and process noise. The first part is straight forward, so we will focus on the second part here. Actually many people noticed there are lots of mystery selected\_texts during the competition. For example:
```
[text]           : [ hey mia!  totally adore your music.  when will your cd be out?]
[selected_text]  : [y adore]
```
Take Method-1 as an example, it throws [y] away, leaving [ adore] in the target span. Therefore, the trained model is good at predicting [ adore] as the results (jaccard(decode_target, pred) = 1), but due to the noise [y], the final jaccard is only 0.5. In fact, for a model with validation jaccard score around 0.717, the jaccard score between prediction and the decode target is around 0.724. Hence if we can somehow add back the noise, we can boost model performance. For the above expample, one may naturelly think the original label is [ adore ], therefore the given label can be acheived by shifting the span to the left by one position. Later, we realized how these shifts come from, if we compare the text and the clean text,

```
[text]           : [ hey mia!  totally adore your music.  when will your cd be out?]
[clean text]     : [hey mia! totally adore your music. when will your cd be out?]
```
You will realize there are 3 extra spaces in the text, one leading and two in between. Ignore the leading space and the one after selected_text span, there is one extra space, which is exactly the number of shift position.
```
[text_no_leading]: [hey mia!  totally adore your music.  when will your cd be out?]
[selected text]  :                 [y adore]
[clean text]     : [hey mia! totally adore your music. when will your cd be out?]
[selected text]  :                 [ adore ]
```
We think the stroy behind is that the label provider only provided the indices of targets to Kaggle ( @philculliton mentioned it [here](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/140847#802761)). There might be some miss communication between Kaggle and the service provider regarding which texts they are using (Kaggle uses raw text, labeler uses clean text). 

Based on this finding, we designed rules to post-process model predictions and clean labels. As a results, both models could predict the **ground truth** start index accurately, while the starting noise is handled by post processing. The post processing rules are,
```python
def _post_shift_new(text, pred):
    """
    text: raw text
    pred: prediction based on raw text, no leading space
    """
    clean_text = " ".join(text.split())
    start_clean = clean_text.find(" ".join(pred.split()))
    
    start = text.find(pred)
    end = start + len(pred)
    extra_space = start - start_clean 
    
    if start&gt;extra_space and extra_space&gt;0:
        if extra_space==1:
            if text[start-1] in [',','.','?','!'] and text[start-2]!=' ':
                start -= 1
        elif extra_space==2:
            start -= extra_space
            if text[end-1] in [',','.','!','?','*']:
                end -= 1
        else:
            end -= (extra_space-2)
            start -= extra_space
    pred = text[start:end]
    
    # handle single quotation mark
    if pred.count("'") == 1:
        if pred[0] == "'":
            if text.find(pred) + len(pred) &lt; len(text) and text[text.find(pred) + len(pred)] == "'":
                pred += "'"
        else:
            if text.find(pred) - 1 &gt;= 0 and text[text.find(pred) - 1] == "'":
                pred = "'" + pred               
    return pred
```
We applied post processing to all the non-neutral samples, for all those modified samples, it has a wining rate (jac\_post &gt; jac\_no\_post) of ~60% on the validation set. Our best single achieves CV 0.7264, LB 0.728 after post processing. So processing the noise brings ~0.009 boost in both CV and LB. The winning rate and aligned boost in both CV &amp; LB make us feel comfortable to apply it. 

# Ensemble
To ensemble results from different models, we convert the token probability to char-level probability, ensemble the probabilities from different models and finally find the start/end index. By ensembling 4 models (2 patterns X 2 models) and applying post-processing, we achieved CV 0.7299, LB 0.723, which scores 0.730 on private leaderboard. 

# Other stuff
* Actually we didn't select our best solution, which is an ensemble of 3 RoBERTa-base and a RoBERTa-large, it has a CV 0.730669, LB 0.721 and private LB 0.731. 
* I checked all our submissions with private leaderboard score, CV aligns perfectly with private leaderboard. Never forget the golden rule: **Trust your CV** 
