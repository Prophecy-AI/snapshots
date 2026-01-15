# 13th Place - From LB 710 to LB 726 - How To

**Rank:** 13
**Author:** Chris Deotte
**Collaborators:** Chris Deotte, Shai, Md Yasin Kabir, cooleel
**Votes:** 116

---

Thank you to my wonderful teammates Shai @sgalib, Yasin @mykttu, Cooleel @cooleel . Together we achieved Gold and I am so excited to receive my 5th competition Gold and become **Competition Grandmaster** ! 

Also congrats to Shai @sgalib for becoming Competition Grandmaster. And Cooleel @cooleel for becoming Competition Master!

# Single TensorFlow Roberta-base
Our best final submission is a single TensorFlow RoBERTa model. We start with my public notebook [here][1] which has CV 0.705, Public LB 0.709, Private LB 0.713. Then we make 10 changes to increase to CV 0.718, Public LB 0.724, Private LB 0.726

# Validation
We tested dozens, maybe hundreds of ideas. Since training data was small, for each idea, we ran the local CV 10 times with 10 different K Fold random seeds and averaged the scores (that's 5 folds times 10 equals 50). Each change below increased CV average by at least 0.001

# 1. Do not remove extra white space.
The extra white space contains signal. For example if text is `"that's awesome!"` then selected text is `awesome`. However if text is `"  that's awesome!"` then selected text is `s awesome`. The second example has extra white space in the beginning of text. And resultantly the selected text has an extra proceeding letter.

# 2. Break apart common single tokens
RoBERTa makes a single token for `"..."`, so your model cannot chose `"fun."` if the text is `"This is fun..."`. So during preprocess, convert all single `[...]` tokens into three `[.][.][.]` tokens. Similarily, split `"..", "!!", "!!!"`. 

# 3. Underestimate train targets
Jaccard score is higher is you underestimate versus overestimate. Therefore if text is `"  Matt loves ice cream"` and the selected text is `"t love"`. Then train your model with selected text `"love"` not selected text `"Matt love"`. All public notebook do the later, we suggest the former.

# 4. Modified Question Answer head
First predict the end index. Then concatenate the end index logits with RoBERTa last hidden layer to predict the start index.

    # ROBERTA
    bert_model = TFRobertaModel.from_pretrained('roberta-base')
    x = bert_model(q_id,attention_mask=q_mask,token_type_ids=q_type)

    # END INDEX HEAD
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2b = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2b)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    # START INDEX HEAD
    x1 = tf.keras.layers.Concatenate()([x2b,x[0]])
    x1 = tf.keras.layers.Dropout(0.1)(x1) 
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)

    # MODEL
    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_type], outputs=[x1,x2])

# 5. Use label smoothing

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)

# 6. Mask words
Use data loader to randomly replace 5% of words with `[mask]` token 50264. Within your dataloader use the following code. We also maintain where the special tokens are so that they don't get replaced

    r = np.random.uniform(0,1,ids.shape)
    ids[r&lt;0.05] = 50264 
    ids[tru] = self.ids[indexes][tru]

# 7. Decay learning rate

    def lrfn(epoch):
        dd = {0:4e-5,1:2e-5,2:1e-5,3:5e-6,4:2.5e-6}
        return dd[epoch]
    lr = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

#8. Train each fold 100% data for submit
After using normal 5 fold and early stopping, note how many epochs are optimal. Then for your LB submission, run your 5 folds with the fixed epoch number you found using 100% data each fold.

#9. Sample weight positive and negative
In TensorFlow Keras it is easy to make certain training samples more important. The normal output from `class DataGenerator(tf.keras.utils.Sequence)` is `(X,y)`. Instead output `(X,y,w)` where weight is the same shape as `y`. Then make `w=2` for all the positive and negative targets and `w=1` for all the neutral targets. Then train with the usual TensorFlow Keras calls

    t_gen = DataGenerator()
    model.fit(t_gen)

And volia! CV and LB increase 0.001

#10. Post process
The above 9 changes already predict much of the noise. For example the above has no problem with the following 2 examples. Text is `"  that's awesome!!!"` with selected text `"s awesome!"`. And `"  I'm thinking... wonderful."` with selected text `". wonderful"`. In each case, the model sees the leading double white space and extracts the single proceeding character.

However the model cannot break a single letter off a word like text `"went fishing and loved it"` with selected text `"d loved"`. This would require breaking a `"d"` off of the word `"and"`. For these difficult cases, we use post process which increase CV 0.0025 and LB 0.0025

    # INPUT s=predicted, t=text, ex=sentiment
    # OUTPUT predicted with PP

    def applyPP(s,t,ex):
    
        t1 = t.lower()
        t2 = s.lower()

        # CLEAN PREDICTED
        b = 0
        if len(t2)&gt;=1:
            if t2[0]==' ': 
                b = 1
                t2 = t2[1:]
        x = t1.find(t2)
        
        # PREDICTED MUST BE SUBSET OF TEXT
        if x==-1:
            print('CANT FIND',k,x)
            print(t1)
            print(t2)
            return s
                
        # ADJUST FOR EXTRA WHITE SPACE
        p = np.sum( np.array(t1[:x].split(' '))=='' )
        if (p&gt;2): 
            d = 0; f = 0
            if p&gt;3: 
                d=p-3
            return t1[x-1-b-d:x+len(t2)]
    
        # CLEAN BAD PREDICTIONS
        if (len(t2)&lt;=2)|(ex=='neutral'):
            return t1
    
        return s

# Other ideas

Our team tried tons of more ideas which may have worked if we spent more time to refine them. Below are some interesting things we tried:
* replacing `****` with the original curse word.
* using part of speech information as an additional feature
* using NER model predictions as additional features
* compare test text with train text using Jaccard and use train selected text when `jac &gt;= 0.85` and `text length &gt;= 4` . (This gained 0.001 on public LB but didn't change private LB).
* pretrain with Sentiment140 dataset as MLM (masked language model) 
* pseudo label Sentiment140 dataset and pretrain as QA (question answer model)
* Train a BERT to choose the best prediction from multiple BERT predictions.
* Stack BERTs. Append output from one BERT to the QA training data of another BERT.
* Tons of ensembling ideas like Jaccard expectation, softmax manipulations, voting ensembles, etc

# Thank you
Once again, thank you to my wonderful teammates @sgalib @mykttu @cooleel . And thank you to Kaggle for another fun competition.


[1]: https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705
