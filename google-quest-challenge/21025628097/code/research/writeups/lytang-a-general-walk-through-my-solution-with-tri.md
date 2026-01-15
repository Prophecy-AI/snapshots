# A general walk through my solution with trick 55th

**Rank:** 55
**Author:** Liyan Tang
**Collaborators:** Liyan Tang, FeZerd, SuperHandsome
**Votes:** 16

---

Congratulations to all winners of this competition. Your hard work paid off!

First, I have to say thanks to the authors of the following three published notebooks:
[https://www.kaggle.com/akensert/bert-base-tf2-0-now-huggingface-transformer](https://www.kaggle.com/akensert/bert-base-tf2-0-now-huggingface-transformer),
[https://www.kaggle.com/abhishek/distilbert-use-features-oof](https://www.kaggle.com/abhishek/distilbert-use-features-oof),
[https://www.kaggle.com/codename007/start-from-here-quest-complete-eda-fe](https://www.kaggle.com/codename007/start-from-here-quest-complete-eda-fe).

These notebooks showed awesome ways to build models, visualize the dataset and extract features from non-text data. 

Our initial plan was to take *question title*, *question body* and *answer* all into a Bert based model. But after we analyzed the distribution of the lengths of question bodies and answers, we found two major problems:
          1. If we fitted all three parts as input, we had to adjust the input space for both *question body* and *answer* due to the limitation of the input size of the Bert based models. In order to do this, we had to trim a bunch of text, which was a waste of training data. Also, the question of how to trim a long text immediately presented itself to us. 
           2. Roughly half of the question bodies and answers had code in them. When implementing tokenization, it really brought some troubles for us. The tokenization of the code looked extremely wired, and we didn’t know if these would have a large effect on the model prediction.

We later did some experiments (I will list some at the end of the post and these turned out to not help a lot) and finally decided to train two separate models. The first one was to fit Bert based models with only *question title* and *question body*. The second one was to fit with only *question title* and *answer*. In this way, almost all question bodies and answers can be fitted into models. 

Like other authors in the discussion, we split 30 output features into two parts. We used the *question title* and *question body* to predict features related to questions and used the *question title* and *answer* to predict features related to answers. We trained weights for several times, and the final output is the average of several models with these weights. 

After doing all of these, we received a 0.415 on the public LB (ranked around 90 at that time). For the rest of the days, we shifted our focus to the data post-processing part cause at that time with the highest scores being over 0.480. But we didn’t believe the models we used had such a big difference. There must have been some trick parts that we had not come to realize. 

So, I checked and compared the distribution of the model prediction and the standard distribution of the training set. We had then found out the “magic” part: the ratings were DISCRETE in the training set. It makes sense, right? Probably raters could only choose certain scores that “make sense”. No one is going to give a score like 0.76354237 in real life :). So, I just did a simple adjustment: pick out every prediction output, find the closest score that “makes sense”, and then replace it. That simple change raised my scores and rank dramatically (LB over 0.450).

I then plotted the distribution of prediction and the standard distribution of the training set again and fine-tuned (actually did some hard coding) every column. This process took me about two days. I submitted many versions, but it did not turn out well with the risk of overfitting. Finally, I selected one kernel with no fine-tuning and one with fine-tuning (hard coding) as my final submission, and the one with no fine-tuning won out.


Things that didn’t work:
          1. Since humans can still understand a sentence even if it missed some words, I tried to define a function that can randomly drop some words from the text and then make it compatible with the input. This did increase my public LB score around 0.003-0.004, but after we decided to build a model with separate two parts, it had little effect.
          2. I roughly went through the text of the dataset and tried to find something in common among texts that have codes (It spent me almost a whole day &gt;_&lt; ). I then used some regular expressions and tried to delete all codes and links in all texts. However, I am not entirely sure if it was good to delete them. Creating some special tokens might be a better idea, but we really didn’t have enough time to do that. It turned out to not work that well and even made the score lower, so we decided to not use it. 

##UPDATE:
Here is an example when I tried to compare the distribution of the training dataset and the prediction.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1801021%2F496a430d255033e9002f4a38e7b9b828%2Fexample%20of%20dsitribution.png?generation=1581441524382191&amp;alt=media)

In the following image, I was trying to mark which columns I need to do fine-tuning. The horizontal yellow lines indicated that I didn't need to do too much about those columns. The scores like 0.5 and 1.0 in other graphs indicated that I need to do some extra thresholding at those values.  

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1801021%2F13ff762ee5a47a3461971b82e3351f10%2Fstandard.png?generation=1581441885285144&amp;alt=media)



##Update Technical Part:

### **Feature Engineering:**
Inspired by pubic kernels, we extract URLs from the *url* column and implemented one-hot
encoding to both the URLs and category columns. There are 64 new feature columns (51 for
*url* and 13 for *category*) after one-hot encoding. According to the paper *Universal Sentence
Encoder*, The Universal Sentence Encoder (USE) encodes text into high dimensional vectors
that can be used for diverse tasks. The input is the variable-length English text, and the output
is a 512-dimensional vector. So, we encoded question title, question body, and answer into
512-dimensional vectors and then compared the L2 distance and cosine similarities between
them. We then added all of these as new features. There are 512 + 512 + 64 + 2 = 1090 new
features.
Under the concern of overfitting, I tried to decrease the number of generated features by
briefly reading through a few examples of each category and then manually reclassifying 51
categories into four sub-categories: *have code, daily life, academic, literature readings*.
However, this didn’t improve the prediction.


###**Model Structure:**
We used a combo of two pretrained Roberta base models to train the dataset. The first
Roberta handles the question title and question body pairs, while the second Roberta handles
the question title and answer pairs. We only fitted texts into Roberta models and ignored
1090 new features for now.
The Roberta base model has 12 hidden layers and there are 13 layers in total (which includes
the final output layer). Every layer has the output dimension *batch_size* x 512 x 768 (*batch
_size x max_seq_len x emb_size*). Initially, I tried out convolutions with different filter sizes
such as 4 x 768, 6 x 768 or 8 x 768 to capture the meaning of the combination of words.
However, it turned out to not work that well, and the average of the whole sequences
outperformed them.
We took the last three output layers out, concatenated them together and then applied an
average-pooling on it. After this, we averaged over 512 tokens for each input in the batch to
capture the meaning of the whole sequence and hoped the model would learn some lower
level features and added them to the final representation. Finally, we concatenated 1090
features with the average embedding and then added a fully connected layer with 21 units for
the title-body pair model and 9 units for the title-answer pair model.
I didn’t use the first unit of the output layer of the Roberta model since Roberta removes the
*Next Sentence Prediction (NSP)* task from BERT’s pre-training, and the first unit seems not
as useful as in Bert. The final predictions are the average of 6 pairs of Roberta models (6-
fold).
We were supposed to freeze Roberta models and then fine-tune the added fully connected
layer, but we didn’t due to the time limit.


###**Customized Learning rate:**
We used a range test in the beginning, to find out the proper learning rate during training. We
then wrote up a customized scheduler inherited from PolynomialDecay to change the
learning rate dynamically. We set up warm-up steps proportional to the total decay-steps to
avoid the primacy effect at the beginning of the training process. The learning rate increases
linearly over the warm-up period and we didn’t set up the learning rate to have cyclic
behavior.


###**Training:**
Like most of the other groups, we use GroupKFold with n_splits=5, did 6-fold cv for 6
epochs and saved the training weights with the highest cv scores. We used Adam optimizer
with mixed-precision data types, which dynamically and automatically adjusting the scaling
to prevent Inf or NaN values and saved training time.


###**Post-processing:**
We first considered using one-hot encoding for all output columns, given that every column
has finite discrete values, but this shouldn’t work well since it didn’t take the order of the
values into account. Because of the time limit, we just compared the distribution of
predictions for each column with the distribution of actual values in the training set and then
adjust the threshold based on the distribution. We compared the scores between before post-
processing and after post-processing to see if it improved performance.