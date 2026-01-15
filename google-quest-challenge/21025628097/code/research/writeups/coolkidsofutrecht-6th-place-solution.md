# 6th place solution

**Rank:** 6
**Author:** Robin N
**Collaborators:** Jing Qin, Zhe Sun, Ahmet Erdem, Robin N
**Votes:** 30

---

Thank you Kaggle and hosts for providing us with this challenging competition and congratulations to all winners! Also I'd like to thank my teammates @aerdem4, @zhesun and @jingqinnl for their hard work and great insights. This was my first time teaming up and definitely made the competition process all the more enjoyable for me.

In short, our final submission is based on a weighted blend of 4 Siamese/double transformer architectures and one USE + feature engineering model coupled with a rounding based post-processing approach.

### 1. Post-processing/Ensembling
I'll start of describing our post-processing strategy, given that as for many other teams this had a massive impact on our performance. In combination with weighted ensembling it improved our 10 fold GroupKFold CV by ~0.05. The general idea is based on rounding predictions downwards to a multiple of some fraction `1/d`: 
```
def scale(x, d):
    if d:
        return (x//(1/d))/d
    return x
```
So if `d=4` and `x = [0.12, 0.3, 0.31, 0.24,  0.7]` these values will get rounded to  `[0.0, 0.25, 0.25, 0.0, 0.5]`. For each target column we did a grid search for values of `d` in `[4, 8, 16, 32, 64, None]`. 

In our ensembling we exploited this technique even further, applying the rounding first to individual model predictions and again after taking a linear combination of model predictions. In doing so we did find that using a separate rounding parameter for each model, OOF score improvements would no longer translate to LB. We addressed this by reducing the number of rounding parameters using the same `d_local` across all models:
```
y_temp = 0
for pred, w in zip(model_preds, ws):
    y_temp += w * scale(pred, d_local) / sum(ws)
y_temp = scale(y_temp, d_global)
```
All ensembling parameters - 2 rounding parameters and 5 model weights - were set using a small grid search optimising the spearman rho metric on OOFs while ignoring question targets for rows with duplicate questions. For all these smart stacking and post-processing tricks the credit goes to @aerdem4. 

### 2. Models
Our final ensemble consists of:
- Siamese Roberta base (CV 0.416)
- Siamese XLNet base  (CV 0.414)
- Double Albert base V2 (CV 0.413)
- Siamese Bert base uncased (CV 0.410)
- USE + Feature Engineering model (CV 0.393)

Listed CV scores are 10 fold GroupKFold w/o post-processing. Although, the transformer models scored significantly higher in terms of CV, the USE + Feature Engineering still contributed significantly in the stack (about 0.005 boost on CV and LB).

All transformer models were implemented using Pytorch and used the pretrained models from the huggingface Transformers library as backbones. Transformer models were trained locally on one RTX 2080Ti. The USE + feature engineering model was implemented with Keras and trained using Kaggle kernels (code available here [https://www.kaggle.com/aerdem4/qa-use-save-model-weights](https://www.kaggle.com/aerdem4/qa-use-save-model-weights). As this model was developed by my teammates I will rely on them to provide more details regarding features, architecture and training in the comment section if needed. 

Apart from the pretrained backbones all transformer architectures were very similar:
- `question_title` + `question_body` and `question_title` + `answer` are fed separately as input to a transformer. As for other top teams, this was easily the biggest difference maker in terms of architecture, adding up to 0.01 to CV scores.
- Average pooling. This improved CV for some models (~ 0.002), but was similar to CLS output for other models.
- Custom 2 layer deep regression head also taking one hot encoded category feature as input. Improved CV ~0.005 relative to simpler linear regression heads.

Only difference between the 4 transformer architectures is that Roberta, XLNet and Bert all used a Siamese design - i.e. the same transformer (shared weights) is used for both question and answer inputs. For Albert using a separate transformer (non-shared weights) worked better. 

### 3. Training
The training followed the exact same format for all four transformers and consisted of 2 stages.

##### First stage:
- Train for 4 epochs with huggingface AdamW optimiser.
- Binary cross-entropy loss.
- One-cycle LR schedule. Uses cosine warmup, followed by cosine decay, whilst having a mirrored schedule for momentum (i.e. cosine decay followed by cosine warmup). 
- Max LR of 1e-3 for the regression head, max LR of 1e-5 for transformer backbones.
- Accumulated batch size of 8

##### Second stage:
Freeze transformer backbone and fine-tune the regression head for an additional 5 epochs with constant LR of 1e-5. Added about 0.002 to CV for most models.

### 4. Pre-processing
No special pre-processing on text inputs, just the default model specific tokenisers provided by huggingface. For target variables we did find a trick. First rank transform and follow up with min-max scaling. This made it so that target values were much more evenly distributed between 0 and 1, which played well with BCE loss. Gave 0.003 - 0.005 boost on CV.


Code for training the transformer models and post-processing/ensembling is available here: [https://github.com/robinniesert/kaggle-google-quest](https://github.com/robinniesert/kaggle-google-quest)