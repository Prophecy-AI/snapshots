# 1st place solution with code

**Author:** Oleg Yaroshevskiy
**Rank:** 1
**Votes:** 328

---

Congratulations to all teams and especially to daring solo players! Thanks to organizers as well!

Tremendous feeling for the whole team. I happened to finish one place away from the golden zone in 2 NLP competitions, and there it is. A few posts below I wrote down "respect your efforts, not results" and you'll be there. There is always a room for such motivational quotes. I don't know whether it's harder to compete with yourself or with such great minds. Thanks to the team ( @ddanevskiy , @kashnitsky , @dmitriyab ) for this thrilling race. And thanks to the whole community for ideas, criticism, and a sense of humor. 


**EDIT**

The community seems to be waiting for a killer feature from us. I don’t reckon we’ve got a single one. Our solution is comprised of three cornerstones:
- pretrained language models
- pseudo-labeling
- post-processing (yes we did one)


**The basic pipeline**

Less than 3 weeks ago just when we were done with the TF Q&amp;A competition, we jumped into this one. Oh yes, there were already so many great NLP folks seen on the leaderboard. We started from a public pytorch baseline [(0.377 I guess)] (https://www.kaggle.com/phoenix9032/pytorch-bert-plain) (kudos @phoenix9032) and managed to update it effectively (0.396 for BERT-base and 0.402 for BERT-large) with a couple of tricks:
- GroupKFold with question_title groups
- Multi-Sample Dropout for Accelerated Training and Better Generalization, [paper](https://arxiv.org/pdf/1905.09788.pdf)
- Different learning rate settings for encoder and head (deserves a paper @ddanevskyi)
- Using CLS outputs from all BERT layers rather than using only the last one. We did this by using a weighted sum of these outputs, where the weights were learnable and constrained to be positive and sum to 1

We experimented with separate weights for different targets and with various loss functions but eventually ended up sticking to simple BCE loss. 


**Pretrained language models**

When @dmitriyab joined our team, he introduced stackexchange (SE) pretrained LMs with 110k-long dictionary including code and other domain tokens. In addition to the common MLM task, the model also predicted 6 auxiliary targets (`question_score, question_view_count, question_favorite_count, answer_score, answers_count, is_answer_accepted`) with a linear layer on top of pooled LM output. That looked beneficial enough for me to drop my code replacement experiments and update the existing bert pipeline with a huge pretrained embedding layer -&gt; 0.414. In the end @dmitriyab and @kashnitsky trained a few LMs (bert-large, roberta, roberta-large) on 7M SE data using whole-word-masking, but the initial model was still the best. 


**Pseudo-labeling**

Pseudo-labeling is a well known semi-supervised learning technique within the Kaggle community and sometimes I wonder how they reinvent stuff in academic papers like [this one](https://arxiv.org/pdf/1911.04252.pdf) (btw a great result). So we labeled additional 100k samples using our three basic bert models. First experiments with pseudo-labels showed huge improvement for SE pretrained bert (0.414 -&gt; 0.445). That was huge. But the leaderboard did not agree.
 
Previously, my teammates and I used models from all folds to create pseudolabels. This strategy worked well in many cases including recent Recursion competition, where my team finished 6th. @ddanevskyi used the same approach for his 3rd place on Freesound. This way of creating pseudolabels gives quite accurate predictions as you are using a fold ensemble, but has an unobvious pitfall. If the pseudolabeled set contains a lot of similar samples to the training set, you could expect that some actual training labels could leak through those samples, even if there is no “direct” target leakage. 

Imagine that you’ve trained a 5-fold model on the training set then used those 5 folds to create pseudolabels on some external dataset. Then you add those pseudolabels to the training part of each of 5 folds. Very likely you will see an overoptimistic score on the validation part as 4 out of 5 models used to generate pseudolabels have “seen” the current validation set through their training process.

We could expect a lot of very similar questions/answers in SO&amp;SE data, so we clearly couldn’t rely on that sort of pseudolabels. In order to fix the problem, we generated 5 different sets of pseudolabels where for each train/val split we used only those models that were trained using only the current train set. This gave much less accurate pseudolabels, but eliminated any possible source of target leakage.

As expected, it didn’t bring such an improvement compared to a “leaked” approach (0.414 -&gt; 0.422) but that was a wise decision in terms of tackling overfitting. 


**BART**

One week prior to the deadline, we noticed @christofhenkel ’s post in the external data thread - the one about [fairseq BART](https://github.com/pytorch/fairseq/tree/master/examples/bart) . Autoregressive sequence-to-sequence model for text classification? Looked strange to me but at the same time that was something new and daring, and we wouldn’t ignore a hint by such a prominent kaggler. It took us three whole days to train BART on 1080ti with batch size 2. The model scored 0.427 locally and it’s predictions decorrelated well with those of BERT. 

We also trained a few other bert-base, bert-large, roberta-base, and roberta-large models. They all scored around 0.420 locally but didn’t improve drastically our bert+bart baseline. 


**Post-processing**

And finally, post-processing. Yes we did one. Based on [this discussion](https://www.kaggle.com/c/google-quest-challenge/discussion/118724) we found that discretization of predictions for some challenging targets led to better spearman corr. So we used no thresholds but discretized such columns based on their distribution in the training set. That simple heuristics boosted score by 0.027-0.030 for almost every single model (up to 0.448 bert-base, 0.454 bart). 


**Finish**

Our local validation and public score stayed the same for the whole last week - 0.468. Yesterday, since early morning with all this berts, barts, robertas we were not able to reproduce public score :harold:, having only 0.462 submissions but they seem to work the best -&gt; 0.430-0.433 private leaderboard (unfortunately we didn’t select the best one). But that was good enough. 


**What didn’t happen to work well**

- StackExchange meta-features used to boost our score in the very beginning but in the end they made no significant difference
- GPT, GPT2-medium, GPT2-large showed mediocre scores
- backtranslation
- stacking. Instead, linear blends worked unreasonably well

Wonderful teamwork and team patience (not really haha). Thanks to you all. We are making our submission kernels and datasets public. By the end of the day I’m releasing our code on GitHub. 

**EDIT**
Code released [here](https://github.com/oleg-yaroshevskiy/quest_qa_labeling). There are 3 separate branches. I'll update with a nice readme soon. 

*Oleg Yaroshevskiy, Dmitriy Danevskiy, Yury Kashnitsky, Dmitriy Abulkhanov*