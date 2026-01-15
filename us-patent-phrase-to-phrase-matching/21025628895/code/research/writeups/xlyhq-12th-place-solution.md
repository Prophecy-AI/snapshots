# 12th Place Solution

**Rank:** 11
**Author:** yanqiangmiffy
**Collaborators:** yanqiangmiffy, 小白Lan, Leo Lu, heng, wagege
**Votes:** 40

---

## Introduction

In the patent matching dataset, contestants need to judge the similarity of two phrases, one is anchor and the other is target, and then output the similarity between the two in different semantics (context), the range is 0-1.

Our team id is xlyhq, LB rank 13rd, PB rank 12nd, thank you very much `@heng zheng`,` @pythonlan`, `@leolu1998`, `@syzong` . The hard work and dedication of the four teammates finally got the lucky obtain to the gold medal.

Similar to other core ideas in the top teams, here we mainly share our game history and the specific results of related experiments, as well as interesting attempts


## Text processing
The dataset mainly includes anchor, target and context fields, and additional text splicing information. During the competition, we mainly tried the following splicing attempts:

- v1: test['anchor'] + '[SEP]' + test['target'] + '[SEP]' + test['context_text']
- v2: tes- t['anchor'] + '[SEP]' + test['target'] + '[SEP]' +test['context']+ '[SEP]' + test['context_text'], equivalent to Directly splicing A47 similar codes
- v3: test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]' + test['context'] + '[SEP]' + test[ 'context_text'] Get more text for splicing, which is equivalent to splicing the subcategories under A47, such as A47B, A47C

```
context_mapping = {
    "A": "Human Necessities",
    "B": "Operations and Transport",
    "C": "Chemistry and Metallurgy",
    "D": "Textiles",
    "E": "Fixed Constructions",
    "F": "Mechanical Engineering",
    "G": "Physics",
    "H": "Electricity",
    "Y": "Emerging Cross-Sectional Technologies",
}

titles = pd.read_csv('./input/cpc-codes/titles.csv')


def process(text):
    return re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", text)


def get_context(cpc_code):
    cpc_data = titles[(titles['code'].map(len) <= 4) & (titles['code'].str.contains(cpc_code))]
    texts = cpc_data['title'].values.tolist()
    texts = [process(text) for text in texts]
    return ";".join([context_mapping[cpc_code[0]]] + texts)


def get_cpc_texts():
    cpc_texts = dict()
    for code in tqdm(train['context'].unique()):
        cpc_texts[code] = get_context(code)
    return cpc_texts


cpc_texts = get_cpc_texts()
```

This splicing method can be improved a lot, but the text length becomes longer, and the maximum length is set to 300, resulting in slower training

-  v4: The core splicing method: test['text'] = test['text'] + '[SEP]' + test['target_info']
```
# concat target info
test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]' + test['context_text']
target_info = test.groupby(['anchor', 'context'])['target'].agg(list).reset_index()
target_info['target'] = target_info['target'].apply(lambda x: list(set(x)))
target_info['target_info'] = target_info['target'].apply(lambda x: ', '.join(x))
target_info['target_info'].apply(lambda x: len(x.split(', '))).describe()

del target_info['target']
test=test.merge(target_info,on=['anchor','context'],how='left')
test['text'] = test['text'] + '[SEP]' + test['target_info'] 
test.head()

```

This splicing method can greatly improve the cv and lb scores of the model. By comparing the two different splicing methods of v3 and v4, we can find that selecting higher-quality text for splicing can improve the model. The v3 method has A lot of redundant information, and there is a lot of critical information at the entity level in the v4 way.

We were very lucky to find out the “magic trick”, which other golden medal zone teams refer in their solutions, in the final days of the competition.

## CV Split

During the competition, we tried different data partitioning methods, including:

- `StratifiedGroupKFold`: this splicing method has a smaller difference between the cv and lb lines, and the score is slightly better
- `StratifiedKFold`: Offline cv is relatively high
- Other `Kfold` and `GrouFold` do not work well

## Loss function
The main loss functions that can be referred to are:

- BCE: nn.BCEWithLogitsLoss(reduction="mean")
- MSE: nn.MSELoss()
- Mixture Loss: MseCorrloss
```
class CorrLoss(nn.Module):
    """
    use 1 - correlational coefficience between the output of the network and the target as the loss
    input (o, t):
        o: Variable of size (batch_size, 1) output of the network
        t: Variable of size (batch_size, 1) target value
    output (corr):
        corr: Variable of size (1)
    """
    def __init__(self):
        super(CorrLoss, self).__init__()

    def forward(self, o, t):
        assert(o.size() == t.size())
        # calcu z-score for o and t
        o_m = o.mean(dim = 0)
        o_s = o.std(dim = 0)
        o_z = (o - o_m)/o_s

        t_m = t.mean(dim =0)
        t_s = t.std(dim = 0)
        t_z = (t - t_m)/t_s

        # calcu corr between o and t
        tmp = o_z * t_z
        corr = tmp.mean(dim = 0)
        return  1 - corr
    
class MSECorrLoss(nn.Module):
    def __init__(self, p = 1.5):
        super(MSECorrLoss, self).__init__()
        self.p = p
        self.mseLoss = nn.MSELoss()
        self.corrLoss = CorrLoss()

    def forward(self, o, t):
        mse = self.mseLoss(o, t)
        corr = self.corrLoss(o, t)
        loss = mse + self.p * corr
        return loss

```


The loss function used in our experiment is slightly better than BCE

## Models
In order to improve the degree of difference of the models, we mainly selected variants of different models, including the following five models:

- Deberta-v3-large
- Bert-for-patents
- Roberta-large
- Ernie-en-2.0-Large
- Electra-large-discriminator

The specific cv scores are as follows:

```

deberta-v3-large：[0.8494,0.8455,0.8523,0.8458,0.8658] cv 0.85176
bertforpatents [0.8393, 0.8403, 0.8457, 0.8402, 0.8564] cv 0.8444
roberta-large [0.8183,0.8172,0.8203,0.8193,0.8398] cv 0.8233
ernie-large [0.8276,0.8277,0.8251,0.8296,0.8466] cv 0.8310
electra-large [0.8429,0.8309,0.8259,0.8416,0.846] cv 0.8376

```


## Training optimization
According to previous competition experience, we mainly adopted the following model training optimization methods:

- Adversarial training: Tried FGM to improve model training

```
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

```

- Model generalization: added multidroout
- Ema improves model training
```
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
 
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
 
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
 
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
 
# 初始化
ema = EMA(model, 0.999)
ema.register()
 
# 训练过程中，更新完参数后，同步update shadow weights
def train():
    optimizer.step()
    ema.update()
 
# eval前，apply shadow weights；eval之后，恢复原来模型的参数
def evaluate():
    ema.apply_shadow()
    # evaluate
    ema.restore()

```

Tried that didn't work:

- AWP
- PGD

## Model ensemble

According to offline cross-validation scores and online score feedback, we average fusion by weighted fusion：

```
from sklearn.preprocessing import MinMaxScaler
MMscaler = MinMaxScaler()
predictions1 = MMscaler.fit_transform(submission['predictions1'].values.reshape(-1,1)).reshape(-1)
predictions2 = MMscaler.fit_transform(submission['predictions2'].values.reshape(-1,1)).reshape(-1)
predictions3 = MMscaler.fit_transform(submission['predictions3'].values.reshape(-1,1)).reshape(-1)
predictions4 = MMscaler.fit_transform(submission['predictions4'].values.reshape(-1,1)).reshape(-1)
predictions5 = MMscaler.fit_transform(submission['predictions5'].values.reshape(-1,1)).reshape(-1)


# final_predictions=(predictions1+predictions2)/2
# final_predictions=(predictions1+predictions2+predictions3+predictions4+predictions5)/5
# 5：2：1：1：1
final_predictions=0.5*predictions1+0.2*predictions2+0.1*predictions3+0.1*predictions4+0.1*predictions5

```


## Other attempts
- two stage

In the early stage, we made fine-tuning of different pre-training models, so the number of features was relatively large. We tried to stack the text statistical features and model prediction based on the tree model. At that time, the model had a relatively good fusion effect. The following contains some codes

```
# ====================================================
# predictions1
# ====================================================

def get_fold_pred(CFG, path, model):
    CFG.path = path
    CFG.model = model
    CFG.config_path = CFG.path + "config.pth"
    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path)
    test_dataset = TestDataset(CFG, test)

    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    predictions = []
    for fold in CFG.trn_fold:
        model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
        state = torch.load(CFG.path + f"{CFG.model.split('/')[-1]}_fold{fold}_best.pth",
                           map_location=torch.device('cpu'))
        model.load_state_dict(state['model'])
        prediction = inference_fn(test_loader, model, device)
        predictions.append(prediction.flatten())
        del model, state, prediction
        gc.collect()
        torch.cuda.empty_cache()
    # predictions1 = np.mean(predictions, axis=0)

    # fea_df = pd.DataFrame(predictions).T
    # fea_df.columns = [f"{CFG.model.split('/')[-1]}_fold{fold}" for fold in CFG.trn_fold]
    # del test_dataset, test_loader

    return predictions


model_paths = [
    "../input/albert-xxlarge-v2/albert-xxlarge-v2/",
    "../input/bert-large-cased-cv5/bert-large-cased/",
    "../input/deberta-base-cv5/deberta-base/",
    "../input/deberta-v3-base-cv5/deberta-v3-base/",
    "../input/deberta-v3-small/deberta-v3-small/",
    "../input/distilroberta-base/distilroberta-base/",
    "../input/roberta-large/roberta-large/",
    "../input/xlm-roberta-base/xlm-roberta-base/",
    "../input/xlmrobertalarge-cv5/xlm-roberta-large/",
]

print("train.shape, test.shape", train.shape, test.shape)
print("titles.shape", titles.shape)


# for model_path in model_paths:
#     with open(f'{model_path}/oof_df.pkl', "rb") as fh:
#         oof = pickle.load(fh)[['id', 'fold', 'pred']]
# #     oof = pd.read_pickle(f'{model_path}/oof_df.pkl')[['id', 'fold', 'pred']]
#     oof[f"{model_path.split('/')[1]}"] = oof['pred']
#     train = train.merge(oof[['id', f"{model_path.split('/')[1]}"]], how='left', on='id')
    
oof_res=pd.read_csv('../input/train-res/train_oof.csv')

train = train.merge(oof_res, how='left', on='id')

model_infos = {
    'albert-xxlarge-v2': ['../input/albert-xxlarge-v2/albert-xxlarge-v2/', "albert-xxlarge-v2"],
    'bert-large-cased': ['../input/bert-large-cased-cv5/bert-large-cased/', "bert-large-cased"],
    'deberta-base': ['../input/deberta-base-cv5/deberta-base/', "deberta-base"],
    'deberta-v3-base': ['../input/deberta-v3-base-cv5/deberta-v3-base/', "deberta-v3-base"],
    'deberta-v3-small': ['../input/deberta-v3-small/deberta-v3-small/', "deberta-v3-small"],
    'distilroberta-base': ['../input/distilroberta-base/distilroberta-base/', "distilroberta-base"],
    'roberta-large': ['../input/roberta-large/roberta-large/', "roberta-large"],
    'xlm-roberta-base': ['../input/xlm-roberta-base/xlm-roberta-base/', "xlm-roberta-base"],
    'xlm-roberta-large': ['../input/xlmrobertalarge-cv5/xlm-roberta-large/', "xlm-roberta-large"],
}

for model, path_info in model_infos.items():
    print(model)
    model_path, model_name = path_info[0], path_info[1]
    fea_df = get_fold_pred(CFG, model_path, model_name)
    model_infos[model].append(fea_df)
    del model_path, model_name

del oof_res

```

train code:

```
for fold_ in range(5):
    print("Fold:", fold_)

    trn_ = train[train['fold'] != fold_].index
    val_ = train[train['fold'] == fold_].index
#     print(train.iloc[val_].sort_values('id'))
    trn_x, trn_y = train[train_features].iloc[trn_], train['score'].iloc[trn_]
    val_x, val_y = train[train_features].iloc[val_], train['score'].iloc[val_]

    # train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    # valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)

    reg = lgb.LGBMRegressor(**params,n_estimators=1100)
    xgb = XGBRegressor(**xgb_params, n_estimators=1000)
    cat = CatBoostRegressor(iterations=1000,learning_rate=0.03,
                            depth=10,
                            eval_metric='RMSE',
                            random_seed = 42,
                            bagging_temperature = 0.2,
                            od_type='Iter',
                            metric_period = 50,
                            od_wait=20)
    print("-"* 20 + "LightGBM Training" + "-"* 20)
    reg.fit(trn_x, np.log1p(trn_y),eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,verbose=100,eval_metric='rmse')
    print("-"* 20 + "XGboost Training" + "-"* 20)
    xgb.fit(trn_x, np.log1p(trn_y),eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,eval_metric='rmse',verbose=100)
    print("-"* 20 + "Catboost Training" + "-"* 20)
    cat.fit(trn_x, np.log1p(trn_y), eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,use_best_model=True,verbose=100)
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain_reg'] = reg.booster_.feature_importance(importance_type='gain')
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    
    for model, values in model_infos.items():
        test[model] = values[2][fold_]
        
    for model, values in uspppm_model_infos.items():
        test[f"uspppm_{model}"] = values[2][fold_]

        
        
        
#     for f in tqdm(amount_feas, desc="amount_feas 基本聚合特征"):
#         for cate in category_fea:
#             if f != cate:
#                 test['{}_{}_medi'.format(cate, f)] = test.groupby(cate)[f].transform('median')
#                 test['{}_{}_mean'.format(cate, f)] = test.groupby(cate)[f].transform('mean')
#                 test['{}_{}_max'.format(cate, f)] = test.groupby(cate)[f].transform('max')
#                 test['{}_{}_min'.format(cate, f)] = test.groupby(cate)[f].transform('min')
#                 test['{}_{}_std'.format(cate, f)] = test.groupby(cate)[f].transform('std')
            
            
            
    # LightGBM
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
#     oof_reg_preds[oof_reg_preds < 0] = 0
    lgb_preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
#     lgb_preds[lgb_preds < 0] = 0
    
    
    # Xgboost
    oof_reg_preds1[val_] = xgb.predict(val_x)
    oof_reg_preds1[oof_reg_preds1 < 0] = 0
    xgb_preds = xgb.predict(test[train_features])
#     xgb_preds[xgb_preds < 0] = 0
    
    # catboost
    oof_reg_preds2[val_] = cat.predict(val_x)
    oof_reg_preds2[oof_reg_preds2 < 0] = 0
    cat_preds = cat.predict(test[train_features])
    cat_preds[xgb_preds < 0] = 0
        
#     merge all prediction
    merge_pred[val_] = oof_reg_preds[val_] * 0.4 + oof_reg_preds1[val_] * 0.3 +oof_reg_preds2[val_] * 0.3
    
#     sub_reg_preds += np.expm1(_preds) / len(folds)
#     sub_reg_preds += np.expm1(_preds) / len(folds)

    sub_preds += (lgb_preds / 5) * 0.6 + (xgb_preds / 5) * 0.2 + (cat_preds / 5) * 0.2 #三个模型五折测试集预测结果
    
    sub_reg_preds+=lgb_preds / 5 # lgb五折测试集预测结果
print("lgb",pearsonr(train['score'], np.expm1(oof_reg_preds))[0]) # lgb
print("xgb",pearsonr(train['score'], np.expm1(oof_reg_preds1))[0]) # xgb
print("cat",pearsonr(train['score'], np.expm1(oof_reg_preds2))[0]) # cat
print("xgb lgb cat",pearsonr(train['score'], np.expm1(merge_pred))[0]) # xgb lgb cat

```