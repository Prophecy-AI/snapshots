# Quick 3rd place solution (no pp) + source code

**Rank:** 3
**Author:** Khoi Nguyen
**Collaborators:** Khoi Nguyen, Dieter
**Votes:** 72

---

My solution on github: https://github.com/suicao/tweet-extraction/
Will update later when @christofhenkel wakes up, his old body needs that sleep.

The main ingredients for our solution are:
- Character level model with GRU head (Dieter)
- Normal model with beamsearch-like decoder (mine)
- Diversity rules
- RoBERTa base + RoBERTa large + BART large.

We knew what the magic was but couldn't find a reliable post processing method for it. Soon we looked into the predictions and realized that the models did a decent job of overfitting the noise anyway, and focused on that direction. In the end, each of our method scored ~0.728 public LB and could've been in the gold zone.

For my part:
- I used fuzzywuzzy to fuzzy match the labels in case the it was split in half.
- For modeling, I copied XLNet's decoder head for question answering to RoBERTa. Basically you predict the start index, get the *k* hidden states at the *top-k* indices. for each hidden state, concat it to the end index logits and predict the corresponding *top-k* end indices. 
- The best *k* is 3, for whatever reasons, which resulted in a 3x3 start-end pairs. I ranked them  by taking the product of the two probs.
- Nothing too special regarding training pipeline. I used the fastai style freeze-unfreeze scheme since the head is quite complicated.
- Everything with RoBERTa byte-level BPE works best. BERTweet was quite a pain to work with (it also used og BERT's subword BPE for some reason), we didn't include it in the final ensemble.
- Probably for the reason above, XLNet's performance was quite subpar, a shame since it's my favorite.

Updated: added source code.