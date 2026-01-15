# 79th place - Meta Feature Analysis

**Rank:** 79
**Author:** Chris Deotte
**Collaborators:** Chris Deotte
**Votes:** 41

---

Thank you everyone for sharing your great solutions. I'm learning so much!

I had only a few days to work on this competition and I didn't know anything about NLP. So I spent a few days reading about BERT and a few days exploring meta features and analyzing the Spearman's rank correlation metric. 

I learned a lot from Akensert's great public kernel [here][1]. If you start with that and incorporate meta features and apply a Spearman rank transformation, you can achieve one position short of a Silver medal a.k.a The Best Bronze Solution :-)

# Meta Features
Consider the following meta features
* `question_user_name`
* `question_title_word_count`, `question_title_first_word`, `question_title_last_word`
* `question_body_word_count`, `question_body_first_word`, `question_body_last_word`
* `answer_user_name`
* `answer_word_count`, `answer_first_word`, `answer_last_word`
* `category`, `host`
* frequency encode all the above

If we build LGBM models for each target using these 26 meta features and display the top 3 feature importances, we discover some interesting relationships. Above each image below is the target name and LGBM Spearman's rank correlation validation score:

## Question Body Critical

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1723677%2F273250fcf82c1639aa2504e0ab874d50%2Fbody_crit.png?generation=1581440405894463&amp;alt=media)

Question body word count helps predict `question_body_critical`. That makes sense.

## Question Interesting Others

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1723677%2Fa9828b713cfe7aaf7bd3c4e6649ed813%2Finteresting_others.png?generation=1581440530534709&amp;alt=media)

Meta feature `host` helps predict whether a question is interesting to others. It's amazing that these meta features can achieve a score of 0.314 which is better than public BERT notebooks! Let's see how good `host` is by itself:

    train, validate = train_test_split(train, test_size=0.2, random_state=42)
    mm = train.groupby('host')['question_interestingness_others'].mean().to_dict()
    validate['cat'] = validate.host.map(mm)
    spearmanr(validate.question_interestingness_others,validate.cat).correlation
    # This prints 0.311

This code shows that we can build a model from just the feature host and achieve a validation score of 0.311 predicting `question_interestingness_others` ! Wow

## Question Interesting Self

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1723677%2Fee6342f07a2ee54aa08cc170746098cc%2Finteresting_self.png?generation=1581441808653553&amp;alt=media)

Once again, the meta feature `host` does a great job predicting `question_interesting_self`. By itself it scores 0.442 which does as good as public notebook BERTs!

## Question Type Spelling

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1723677%2F520ced1e879a9661265024c4b9dda477%2Fspelling.png?generation=1581441968092095&amp;alt=media)

LGBM says that the most important feature to predict `question_type_spelling` is `answer_word_count`. From public kernels, we know that `host=='ell.stackexchange.com'` and `host=='english.stackexchange.com'` are very predictive too. (LGBM didn't find this which shows that LGBM is not optimizing Spearman's rank correlation). Using `host` and `answer_word_count`, we can predict `question_type_spelling` with score 0.300 ! Wow

    validate['h_type'] = validate.host=='ell.stackexchange.com'
    validate['h_type'] |= validate.host=='english.stackexchange.com'
    validate['h_type'] &amp;= validate.answer.map(lambda x: len(x.split()))&gt;95
    spearmanr(validate.question_type_spelling,validate.h_type).correlation
    # This prints 0.300

## Question Type Definition

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1723677%2F0dfbb4bf8645dd9f043f7a61f751fcfa%2Fdefinition.png?generation=1581443385397518&amp;alt=media)

Question first word helps predict whether we have `question_type_definition`.

## Question Well Written

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1723677%2F69285f93b55a135fd87ba01f1d264754%2Fwell_written.png?generation=1581442904056045&amp;alt=media)

Question first word frequency helps predict whether a question is well written. That's interesting. If your writing is similar to how other people write, it is deemed well written.

## Answer Level of Information

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1723677%2Fd2b111e2e208ff739dbce612b5444f5d%2Finfo.png?generation=1581443008426365&amp;alt=media)

Answer word count helps predict `answer_level_of_information`. That makes sense. The code below shows that this single feature alone can predict better than public BERT notebooks!

    validate['a_ct'] = validate.answer.map(lambda x: len(x.split()))
    spearmanr(validate.answer_level_of_information,validate.a_ct).correlation
    # This prints 0.392

## Answer Plausible
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1723677%2Fbad5aaee8d43f49f3a9282c3381ae25f%2Fplausible.png?generation=1581443262147882&amp;alt=media)
 
Answer user name frequency helps predict whether an answer is plausible. This makes sense. Active users are probably more plausible.

# Meta Features Applied
Now that we know important meta features, we can either incorporate them into our BERT model inputs, or we can ensemble them with BERT's outputs. Using my OOF, I ensembled meta features to my BERT outputs. For example, the two best CV/LB increases came from replacing BERT's outputs for `question_type_spelling` with 

    test['a_ct'] = test.answer.map(lambda x: len(x.split()))
    test['h_type'] = test.host=='ell.stackexchange.com'
    test['h_type'] |= test.host=='english.stackexchange.com'
    test['h_type'] &amp;= test.a_ct&gt;95
    sub.question_type_spelling = test.h_type.values        

And updating `answer_level_of_information` with:

    sub.answer_level_of_information += 7 * test.a_ct.values/8158
    sub.answer_level_of_information /= sub.answer_level_of_information.max() 

# Spearman's Rank Correlation
As pointed out in this discussion post [here][2], the Spearman's rank correlation metric can be improved if you convert continuous predictions into more discrete predictions if the true targets are discrete. By applying a grid search on my OOF, I found that applying `numpy.clip( pred, clippings[col][0], clippings[col][1] )` can maximize this metric using

    clippings = {
        'question_has_commonly_accepted_answer':[0,0.6],
        'question_conversational':[0.15,1],
        'question_multi_intent':[0.1,1],
        'question_type_choice':[0.1,1],
        'question_type_compare':[0.1,1],
        'question_type_consequence':[0.08,1],
        'question_type_definition':[0.1,1],
        'question_type_entity':[0.13,1]
    }

# UPDATE
During the competition, I only built and trained one `bert-base-uncased` model. Other models didn't have good CVs. But after reading all the great solutions, I just built a `roberta-base` model by first applying `def cln(x): return " ".join(x.split())` which removes white space (as described in 8th place solution [here][3]). 

It turns out that `RobertaTokenizer.from_pretrained('roberta-base')` does not remove white space by itself. (whereas the bert-base-uncased tokenizer does). Using the same architecture as my bert-base, my Roberta has CV 0.395. When I ensemble it with my bert-base CV 0.392, the new CV is 0.412. That's an increase of 0.020 over my bert-base. 

I just submitted to the leaderboard and it increased my LB by 0.015 and my rank by 40 spots! Wow! I just learned how to make better NLP models. I can't wait for the next competition !  

[1]: https://www.kaggle.com/akensert/bert-base-tf2-0-now-huggingface-transformer
[2]: https://www.kaggle.com/c/google-quest-challenge/discussion/118724
[3]: https://www.kaggle.com/c/google-quest-challenge/discussion/129857