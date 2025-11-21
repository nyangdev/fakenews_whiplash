#'epoch=50, batch_size=5000'(임무2)
# 목적: 'content' 내 문장 간 의미 차이가 클수록 가짜 news일 확률이 높다는 가정 하에 'Cosine Similarity Feature + PyTorch TextCNN + FastText Embedding(선택)'을 결합하여 가짜 news 탐지 Model을 학습, 평가, 시각화
import os,random,json,numpy,pandas
import torch,torch.nn as nn,torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
                             roc_auc_score,roc_curve,precision_recall_curve,confusion_matrix)
from gensim.models.fasttext import FastText
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def basic_clean(text:str)->str:
    return str(text).strip()

def tokenize(text:str)->list:
    return simple_preprocess(text,max_len=50)
# 1. Load data
dtrain = pandas.read_csv(r'C:\데이터 엔지니어 부트캠프(4차)\나의 산출물\관련 문서\team Project(박주언, 이창주, 김민지)\가짜 뉴스 탐지 자동화 구현：피해 사전 예방 및 적극적 조치 가능\mission2_train.csv',encoding='utf-8').dropna(subset=['content','label'])
dtest = pandas.read_csv(r'C:\데이터 엔지니어 부트캠프(4차)\나의 산출물\관련 문서\team Project(박주언, 이창주, 김민지)\가짜 뉴스 탐지 자동화 구현：피해 사전 예방 및 적극적 조치 가능\mission2_test.csv',encoding='utf-8').dropna(subset=['content'])

# 2. TF-IDF + cosine_similarity (A)
vec = TfidfVectorizer(max_features=50000,ngram_range=(1,2))
vec.fit(dtrain['content'].astype(str))

tfidtrain = vec.transform(dtrain['content'].astype(str))
tfidtest = vec.transform(dtest['content'].astype(str))

# 대표 TF-IDF 평균 벡터
mean_tfidf = tfidtrain.mean(axis=0)
mean_tfidf = numpy.asarray(mean_tfidf).reshape(1,-1)

# (A) TF-IDF cosine: 각 content vs 전체 평균 content
dtrain['cos_tfidmean'] = [cosine_similarity(tfidtrain[i],mean_tfidf)[0,0] for i in range(tfidtrain.shape[0])]
dtest['cos_tfidmean'] = [cosine_similarity(tfidtest[i],mean_tfidf)[0,0] for i in range(tfidtest.shape[0])]

# (C 확장) TF-IDF self "길이" Feature: 벡터의 L2 norm (자기 유사도 크기 느낌)
dtrain['tfidnorm'] = numpy.sqrt((tfidtrain.power(2)).sum(axis=1)).A1
dtest['tfidnorm'] = numpy.sqrt((tfidtest.power(2)).sum(axis=1)).A1

# 3. FastText 학습 + cosine (B)
all_tokens = [tokenize(basic_clean(c)) for c in dtrain['content']]
ft = FastText(vector_size=100,window=5,min_count=2,workers=4)
ft.build_vocab(corpus_iterable=all_tokens)
ft.train(corpus_iterable=all_tokens,total_examples=len(all_tokens),epochs=50)

def sentence_vec(tokens,ft,dim=100):
    v = numpy.zeros(dim,dtype=numpy.float32); c = 0
    for t in tokens:
        if t in ft.wv:
            v += ft.wv[t]; c += 1
    return v / c if c > 0 else v

mean_vec = numpy.mean([sentence_vec(t,ft,100) for t in all_tokens],axis=0)

def cos(a,b):
    d = numpy.linalg.norm(a) * numpy.linalg.norm(b)
    return float(numpy.dot(a,b) / d) if d > 0 else 0.0

dtrain['cos_ft_mean'] = [cos(sentence_vec(tokenize(c),ft,100),mean_vec) for c in dtrain['content']]
dtest['cos_ft_mean'] = [cos(sentence_vec(tokenize(c),ft,100),mean_vec) for c in dtest['content']]

print('Train shape:',dtrain.shape,' Test shape:',dtest.shape)
# 4. Vocab & numericalization
def build_vocab(texts:list,min_freq:int=2)->dict:
    freq = {}
    for tokens in texts:
        for t in tokens:
            freq[t] = freq.get(t,0) + 1
    itos = ['<pad>','<unk>'] + [t for t,c in sorted(freq.items(),key=lambda x:(-x[1],x[0])) if c >= min_freq]
    stoi = {t:i for i,t in enumerate(itos)}
    return {'itos':itos,'stoi':stoi}

vocab = build_vocab(all_tokens,2)
stoi = vocab['stoi']

def numericalize(tokens,stoi,max_len=256):
    ids = [stoi.get(t,1) for t in tokens][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

# 5. Dataset
class FakeNewsDataset(Dataset):
    def __init__(self,df,stoi,max_len=256):
        self.ids = [numericalize(tokenize(basic_clean(c)),stoi,max_len) for c in df.content]
        # extrdim = 3 → [cos_tfidmean, cos_ft_mean, tfidnorm]
        self.extra = df[['cos_tfidmean','cos_ft_mean','tfidnorm']].astype(numpy.float32).values
        self.labels = df.label.astype(int).values
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        return (torch.tensor(self.ids[idx]),
                torch.tensor(self.extra[idx]),
                torch.tensor(self.labels[idx]))

# 6. TextCNN Model
class TextCNN(nn.Module):
    def __init__(self,vocab,embed_dim,num_classes,filter_sizes=[3,4,5],num_filters=128,dropout=0.5,extrdim=3,pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab['itos']),embed_dim,padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim,num_filters,fs) for fs in filter_sizes])
        self.fc = nn.Linear(num_filters * len(filter_sizes) + extrdim,num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self,ids,extra):
        emb = self.embedding(ids).transpose(1,2)
        feats = [torch.max_pool1d(torch.relu(c(emb)),c(emb).shape[2]).squeeze(2) for c in self.convs]
        out = self.dropout(torch.cat(feats,1))
        return self.fc(torch.cat([out,extra],1))
# 7. Data split & DataLoader
tr_df,val_df = train_test_split(dtrain,test_size=0.1,random_state=42,stratify=dtrain['label'])
tr_ds = FakeNewsDataset(tr_df,stoi)
val_ds = FakeNewsDataset(val_df,stoi)
# test에는 label이 없으므로 dummy 0
te_ds = FakeNewsDataset(dtest.assign(label=0),stoi)

tr_ld = DataLoader(tr_ds,batch_size=5000,shuffle=True)
val_ld = DataLoader(val_ds,batch_size=5000,shuffle=False)
te_ld = DataLoader(te_ds,batch_size=5000,shuffle=False)

# 8. Model, criterion, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextCNN(vocab,100,2,filter_sizes=[3,4,5],num_filters=128,dropout=0.5,extrdim=3,pad_idx=0).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)

# FastText embedding matrix 고정
embed = numpy.zeros((len(vocab['itos']),100),dtype=numpy.float32)
for i,t in enumerate(vocab['itos']):
    if t in ft.wv:
        embed[i] = ft.wv[t]
with torch.no_grad():
    model.embedding.weight.data.copy_(torch.tensor(embed))
    model.embedding.weight.requires_grad = False

hist = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
best = 0.0
model_path = r'C:\cci10000\multi02_datscience\cci10000\fake_FastText_cci10000_2_mission2.pt'

# 9. Training loop (epoch=50)
for e in range(1,51):
    model.train(); loss_sum = correct = total = 0
    for ids,extra,y in tr_ld:
        ids,extra,y = ids.to(device),extra.to(device),y.to(device)
        optimizer.zero_grad()
        logits = model(ids,extra)
        loss = criterion(logits,y)
        loss.backward(); optimizer.step()
        loss_sum += loss.item() * ids.size(0)
        pred = torch.argmax(logits,1)
        correct += (pred == y).sum().item(); total += y.size(0)
    tr_loss = loss_sum / total; tr_acc = correct / total

    # Validation
    model.eval(); val_loss_accum = val_correct = val_total = 0; vp_all = []; vt_all = []
    with torch.no_grad():
        for ids,extra,y in val_ld:
            ids,extra,y = ids.to(device),extra.to(device),y.to(device)
            logits = model(ids,extra)
            loss = criterion(logits,y)
            p = torch.softmax(logits,1)[:,1].cpu().numpy()
            pred = (p >= 0.5).astype(int)
            val_loss_accum += loss.item() * y.size(0)
            val_correct += (torch.tensor(pred) == y.cpu()).sum().item()
            val_total += y.size(0)
            vp_all.extend(p); vt_all.extend(y.cpu().numpy())
    vloss = val_loss_accum / val_total
    vacc = val_correct / val_total
    hist['train_loss'].append(tr_loss); hist['val_loss'].append(vloss)
    hist['train_acc'].append(tr_acc); hist['val_acc'].append(vacc)

    print(f'Epoch {e} : Train Acc={tr_acc:.4f}, Val Acc={vacc:.4f}')
    if vacc > best:
        best = vacc
        torch.save(model.state_dict(),model_path)

print(f'Best Val Acc = {best:.4f}')
vp = numpy.array(vp_all); vt = numpy.array(vt_all)
# 10. Load best model
model.load_state_dict(torch.load(model_path,map_location=device))
model.eval()

# 11. 시각화: Loss/Acc Curve
plt.figure(); plt.plot(hist['train_loss']); plt.plot(hist['val_loss'])
plt.legend(['Train','Val']); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
plt.tight_layout(); plt.savefig(r'C:\cci10000\multi02_datscience\cci10000\mission2_loss_curve.png'); plt.close()

plt.figure(); plt.plot(hist['train_acc']); plt.plot(hist['val_acc'])
plt.legend(['Train','Val']); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.title('Accuracy Curve')
plt.tight_layout(); plt.savefig(r'C:\cci10000\multi02_datscience\cci10000\mission2_acc_curve.png'); plt.close()

# ROC Curve
fpr,tpr,_ = roc_curve(vt,vp)
plt.figure(); plt.plot(fpr,tpr,label=f'ROC AUC={roc_auc_score(vt,vp):.4f}')
plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.title('Validation ROC Curve')
plt.tight_layout(); plt.savefig(r'C:\cci10000\multi02_datscience\cci10000\mission2_val_roc.png'); plt.close()

# PR Curve
prec,rec,_ = precision_recall_curve(vt,vp)
plt.figure(); plt.plot(rec,prec); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Validation Precision-Recall Curve')
plt.tight_layout(); plt.savefig(r'C:\cci10000\multi02_datscience\cci10000\mission2_val_pr.png'); plt.close()

# Confusion Matrix (Validation)
vpred = (vp >= 0.5).astype(int)
cm = confusion_matrix(vt,vpred)
plt.figure(); plt.imshow(cm,interpolation='nearest'); plt.title('Validation Confusion Matrix'); plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,str(cm[i,j]),ha='center',va='center')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout(); plt.savefig(r'C:\cci10000\multi02_datscience\cci10000\mission2_val_confusion.png'); plt.close()

# Cosine TF-IDF 분포 및 label별 평균
plt.figure(figsize=(8,4))
plt.hist(dtrain['cos_tfidmean'],bins=50,alpha=0.7,edgecolor='black')
plt.title('Distribution of TF-IDF Cosine (content vs mean)')
plt.xlabel('cos_tfidmean'); plt.ylabel('Frequency')
plt.tight_layout(); plt.savefig(r'C:\cci10000\multi02_datscience\cci10000\mission2_cos_tfiddistribution.png'); plt.close()

avg_cos = dtrain.groupby('label')['cos_tfidmean'].mean()
print('\n[label별 평균 TF-IDF Cosine]')
for label,mean in avg_cos.items():
    print(f'label={label}→평균 cos_tfidmean={mean:.4f}')

# 12. Test inference
test_probs = []; test_preds = []
with torch.no_grad():
    for ids,extra,_ in te_ld:
        ids,extra = ids.to(device),extra.to(device)
        logits = model(ids,extra)
        p = torch.softmax(logits,1)[:,1].cpu().numpy()
        test_probs.extend(p); test_preds.extend((p >= 0.5).astype(int))

test_probs = numpy.array(test_probs); test_preds = numpy.array(test_preds)

out_df = pandas.DataFrame({
    'content': dtest['content'].values,
    'cos_tfidmean': dtest['cos_tfidmean'].values,
    'cos_ft_mean': dtest['cos_ft_mean'].values,
    'tfidnorm': dtest['tfidnorm'].values,
    'pred': test_preds,
    'prob_fake': test_probs
})
out_path = r'C:\cci10000\multi02_datscience\cci10000\mission2_result_full_cosine.csv'
out_df.to_csv(out_path,index=False,encoding='utf-8')
print('Test 예측 결과 저장 완료→',out_path)