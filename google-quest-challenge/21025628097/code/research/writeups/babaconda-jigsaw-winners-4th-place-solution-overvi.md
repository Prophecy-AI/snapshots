# 4th place solution overview

**Rank:** 4
**Author:** Draconda
**Collaborators:** Draconda, Firas Baba, Ruslan  Talipov, YuryBolkonsky, Nikolay Prokoptsev
**Votes:** 65

---

At first, I want to say thanks for all of my teammates :)

##  Our final submission is an ensemble of 2 models

### About Model

model1 takes 3 texts as input, and has 3berts in it
Bert1.
input is question_title + question_body and predict only columns that are relevant to question (first 21 of columns)
Bert2.
input is question_title + answer and predict only columns that are relevant to answer
(last 9 of columns)
Bert 3.
input is question_body + answer and predict all columns
and have 1 linear layer which input is concat(bert1_out, bert2_out, bert3_out)
and calculate bce for
loss1 for bert1 prediction &amp; columns[:21],
loss2 for bert2 prediction &amp; columns[-9:],
loss3 for bert3 prediction &amp; all columns,
loss4 for last linear layer prediction &amp; all columns
&amp;backward each loss.

model2 is just xlnet version of it.

### About Training

-&gt; Spanish -&gt; English argumentation

Flexible rating module for encoding 2 texts
(ex. if len(text1) &lt; max len1 &amp; Len(text2) &gt; max len2, max len2 = max len2 + (max_len1 - len(text1)))

### About Post Processing

Baba shared the idea here ↓
https://www.kaggle.com/c/google-quest-challenge/discussion/129831

This competition was really fun, thanks all and let's compete at other competition again !!!