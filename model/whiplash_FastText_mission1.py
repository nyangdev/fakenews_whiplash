#'epoch=50, batch_size=5000'(임무1)
# 목적: 'title'과 'content' 간 의미 차이가 클수록 가짜 news일 확률이 높다는 가정 하에 'Cosine Similarity Feature + PyTorch TextCNN + FastText Embedding(선택)'을 결합하여 가짜 news 탐지 Model을 학습, 평가, 시각화
import os,random,json,numpy,pandas
import torch,torch.nn as nn,torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,precision_recall_curve)
from gensim.models.fasttext import FastText
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#전처리 및 tokenize
def basic_clean(text:str)->str:
   return str(text).strip() if pandas.notna(text) else ''
def tokenize(text:str)->list:
   return simple_preprocess(text,max_len=50)

def mkdir(dir:str):
   if not os.path.exists(dir):os.makedirs(dir)

#어휘집 및 Indexing
def build_vocab(texts:list,min_freq:int=2)->dict:
   freq = {}
   for tokens in texts:
      for t in tokens:
         freq[t] = freq.get(t,0) + 1
   itos = ['<pad>','<unk>'] + [t for t,c in sorted(freq.items(),key=lambda x:(-x[1],x[0])) if c >= min_freq]
   stoi = {t:i for i,t in enumerate(itos)}
   return {'itos':itos,'stoi':stoi}

def numericalize(tokens,stoi,max_len=256):
   ids = [stoi.get(t,1) for t in tokens][:max_len]
   ids += [0] * (max_len - len(ids))
   return ids

#FastText 학습 및 Embedding 생성
def train_fasttext(sentences:list,vector_dim=100,epochs=50,workers=4)->FastText:
   model = FastText(vector_size=vector_dim,window=5,min_count=2,workers=workers)
   model.build_vocab(corpus_iterable=sentences)
   model.train(corpus_iterable=sentences,total_examples=len(sentences),epochs=epochs)
   return model
def build_embedding_matrix(vocab,ft_model,vector_dim=100):
   itos = vocab['itos']
   matrix = numpy.zeros((len(itos),vector_dim),dtype=numpy.float32)
   for i,t in enumerate(itos):
     if t in ft_model.wv:matrix[i] = ft_model.wv[t]
   return matrix

#Dataset Class
class FakeNewsDataset(Dataset):
   def __init__(self,df,stoi,max_len=256):
      self.ids = [numericalize(tokenize(basic_clean(t + ' ' + c)),stoi,max_len) for t,c in zip(df.title,df.content)]
      self.cos = df.cos_sim.astype(numpy.float32).values
      self.labels = df.label.astype(int).values
   def __len__(self): return len(self.labels)
   def __getitem__(self,idx):
      return (torch.tensor(self.ids[idx]),torch.tensor([self.cos[idx]]),torch.tensor(self.labels[idx]))

#6. TextCNN Model
class TextCNN(nn.Module):
   def __init__(self,vocab_size,embed_dim,num_classes,filter_sizes=[3,4,5],num_filters=128,dropout=0.5,extrfeat_dim=1,pad_idx=0):
      super().__init__()
      self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
      self.convs = nn.ModuleList([nn.Conv1d(embed_dim,num_filters,fs) for fs in filter_sizes])
      self.fc = nn.Linear(num_filters * len(filter_sizes) + extrfeat_dim,num_classes)
      self.dropout = nn.Dropout(dropout)
   def forward(self,ids,extra):
      emb = self.embedding(ids).transpose(1,2)
      feats = [torch.max_pool1d(torch.relu(c(emb)),c(emb).shape[2]).squeeze(2) for c in self.convs]
      out = self.dropout(torch.cat(feats,1))
      return self.fc(torch.cat([out,extra],1))

#학습, 평가 함수
def train_one_epoch(model,loader,criterion,optimizer,device):
   model.train(); loss_sum = correct = total = 0
   for ids,cos,y in loader:
      ids,cos,y = ids.to(device),cos.to(device),y.to(device)
      optimizer.zero_grad()
      logits = model(ids,cos)
      loss = criterion(logits,y); loss.backward(); optimizer.step()
      loss_sum += loss.item() * ids.size(0)
      pred = torch.argmax(logits,1)
      correct += (pred == y).sum().item(); total += y.size(0)
   return loss_sum / total,correct / total
def eval(model,loader,criterion,device):
   model.eval(); loss_sum = correct = total = 0; probs,preds,trues = [],[],[]
   with torch.no_grad():
      for ids,cos,y in loader:
         ids,cos,y = ids.to(device),cos.to(device),y.to(device)
         logits = model(ids,cos); loss = criterion(logits,y)
         p = torch.softmax(logits,1)[:,1].detach().cpu().numpy()
         pred = (p >= 0.5).astype(int)
         probs.extend(p); preds.extend(pred); trues.extend(y.cpu().numpy())
         loss_sum += loss.item() * y.size(0); correct += (torch.tensor(pred) == y.cpu()).sum().item(); total += y.size(0)
   return loss_sum / total,correct / total,numpy.array(probs),numpy.array(preds),numpy.array(trues)

#시각화 함수들
def plot_roc(y_true,y_prob,path:str):
   fpr,tpr,_ = roc_curve(y_true,y_prob)
   plt.figure(); plt.plot(fpr,tpr,label=f'ROC AUC={roc_auc_score(y_true,y_prob):.4f}')
   plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()
def plot_pr(y_true,y_prob,path:str):
   p,r,_ = precision_recall_curve(y_true,y_prob)
   plt.figure(); plt.plot(r,p); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.tight_layout(); plt.savefig(path); plt.close()
def plot_confusion(cm,path:str):
   plt.figure(); plt.imshow(cm,interpolation='nearest'); plt.title('Confusion Matrix'); plt.colorbar()
   for i in range(cm.shape[0]):
      for j in range(cm.shape[1]): plt.text(j,i,str(cm[i,j]),ha='center',va='center')
   plt.tight_layout(); plt.savefig(path); plt.close()
def plot_history(hist,loss,acc):
   plt.figure(); plt.plot(hist['train_loss']); plt.plot(hist['val_loss']); plt.legend(['Train','Val']); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.savefig(loss); plt.close()
   plt.figure(); plt.plot(hist['train_acc']); plt.plot(hist['val_acc']); plt.legend(['Train','Val']); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.savefig(acc); plt.close()
def plot_cosine_hist(cos_pos,cos_neg,path:str):
   plt.figure(); plt.hist(cos_pos,bins=50,alpha=0.7,label='label=1'); plt.hist(cos_neg,bins=50,alpha=0.7,label='label=0')
   plt.legend(); plt.xlabel('Cosine Similarity'); plt.ylabel('Freq'); plt.savefig(path); plt.close()

# 9. Data를 Load, 실행한 예시
train_path = r'C:\데이터 엔지니어 부트캠프(4차)\나의 산출물\관련 문서\team Project(박주언, 이창주, 김민지)\가짜 뉴스 탐지 자동화 구현：피해 사전 예방 및 적극적 조치 가능\mission1_train.csv'
test_path = r'C:\데이터 엔지니어 부트캠프(4차)\나의 산출물\관련 문서\team Project(박주언, 이창주, 김민지)\가짜 뉴스 탐지 자동화 구현：피해 사전 예방 및 적극적 조치 가능\mission1_test.csv'
dtrain = pandas.read_csv(train_path,encoding='UTF-8')
dtest = pandas.read_csv(test_path,encoding='euc-kr')

#이상치
if 'label' in dtrain.columns:
   invalid_labels = dtrain.loc[~dtrain['label'].isin([0,1])]
   if len(invalid_labels) == 0:
      print('이상치(label) 없음.(정상값은 0 또는 1만 존재)')
   else:
      print(f'이상치(label) : {len(invalid_labels)}건(정상값인 0 또는 1 이외의 값이 존재)')
else:
   print("이상치(label) 확인 불가!('label' Column 없음.)")

#결측치 비율
null_title_train = dtrain['title'].isna().mean() * 100
null_content_train = dtrain['content'].isna().mean() * 100
print(f'결측치 비율(title) : {null_title_train:.2f}%')
print(f'결측치 비율(content) : {null_content_train:.2f}%')

dtrain = dtrain.dropna(subset=['label'])
dtrain = dtrain.dropna(subset=['title'])
dtrain = dtrain.dropna(subset=['content'])

# Cosine Similarity Feature 생성
vec = TfidfVectorizer(max_features=50000,ngram_range=(1,2))
corpus = pandas.concat([dtrain['title'].astype(str),dtrain['content'].astype(str)],axis=0)
vec.fit(corpus)
tfidtitle_train = vec.transform(dtrain['title'].astype(str))
tfidcontent_train = vec.transform(dtrain['content'].astype(str))
dtrain['cos_sim'] = [cosine_similarity(tfidtitle_train[i], tfidcontent_train[i])[0,0] for i in range(tfidtitle_train.shape[0])]

# Test에 대해 transform()을 적용
tfidtitle_test = vec.transform(dtest['title'].astype(str))
tfidcontent_test = vec.transform(dtest['content'].astype(str))
dtest['cos_sim'] = [cosine_similarity(tfidtitle_test[i],tfidcontent_test[i])[0,0] for i in range(tfidtitle_test.shape[0])]

def sentence_vec_fasttest(tokens,ft,dim=100):
   v = numpy.zeros(dim,dtype=numpy.float32); c = 0
   for t in tokens:
      if t in ft.wv:
         v += ft.wv[t]; c += 1
   return v / c if c > 0 else v
def cos_fasttext(title,content,ft,dim=100):
   t = sentence_vec_fasttest(tokenize(basic_clean(title)),ft,dim)
   c = sentence_vec_fasttest(tokenize(basic_clean(content)),ft,dim)
   num = numpy.dot(t,c); denom = numpy.linalg.norm(t) * numpy.linalg.norm(c)
   return float(num / denom) if denom > 0 else 0.0
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark = True,False
dtrain['title'] = dtrain['title'].fillna('').astype(str)
dtrain['content'] = dtrain['content'].fillna('').astype(str)
dtest['title'] = dtest['title'].fillna('').astype(str)
dtest['content'] = dtest['content'].fillna('').astype(str)
all_tokens = [tokenize(basic_clean(t + ' ' + c)) for t,c in zip(dtrain.title,dtrain.content)]
vocab = build_vocab(all_tokens,min_freq=2)
stoi = vocab['stoi']
vocab_size = len(vocab['itos'])
ft = train_fasttext(all_tokens,vector_dim=100,epochs=50)
dtrain['cos_sim_ft'] = [cos_fasttext(t,c,ft,100) for t,c in zip(dtrain['title'],dtrain['content'])]
embed = build_embedding_matrix(vocab,ft,vector_dim=100)
from sklearn.model_selection import train_test_split
tr_df,val_df = train_test_split(dtrain,test_size=0.1,random_state=42,stratify=dtrain['label'])
tr_ds = FakeNewsDataset(tr_df,stoi)
val_ds = FakeNewsDataset(val_df,stoi)
test_ds = FakeNewsDataset(dtest,stoi)
tr_ld = DataLoader(tr_ds,batch_size=500,shuffle=True)
val_ld = DataLoader(val_ds,batch_size=500,shuffle=False)
te_ld = DataLoader(test_ds,batch_size=500,shuffle=False)
model_path = 'fake_FastText_cci10000_2_mission1.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextCNN(vocab_size,100,2,[3,4,5],128,0.5,1,0).to(device)
hist = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
if os.path.exists(model_path):
   print(f"기존 Model '{model_path}' file이 발견되었습니다. Model을 load 합니다.")
   model.load_state_dict(torch.load(model_path,map_location=device))
else:
   print(f"'{model_path}' file이 없습니다. Model을 새로 학습합니다...")
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(),lr=1e-3)
   best=0
   for e in range(1,51):
      tr_loss,tr_acc = train_one_epoch(model,tr_ld,criterion,optimizer,device)
      vloss,vacc,vp,vpr,vt = eval(model,val_ld,criterion,device)
      hist['train_loss'].append(tr_loss); hist['val_loss'].append(vloss)
      hist['train_acc'].append(tr_acc); hist['val_acc'].append(vacc)
      print(f'Epoch {e} : Train Acc = {tr_acc:.4f}, Val Acc = {vacc:.4f}')
      if vacc > best:
         best = vacc; torch.save(model.state_dict(),model_path)
   print(f'Model 학습 완료! 최고 검증 정확도 : {best:.4f}')
   print(f"'{model_path}' file로 저장되었습니다.")
model.load_state_dict(torch.load(model_path,map_location=device))
model.eval()
with torch.no_grad():
   model.embedding.weight.data.copy_(torch.tensor(embed))
   model.embedding.weight.data[0] = torch.zeros(100)
   model.embedding.weight.requires_grad = False
criterion = nn.CrossEntropyLoss()
vloss,vacc,vp,vpr,vt = eval(model,val_ld,criterion,device)

#시각화 예시
plot_history(hist,'loss.png','acc.png')
plot_roc(vt,vp,'roc_val.png')
test_loss,test_acc,test_prob,test_pred,test_true = eval(model,te_ld,criterion,device)

#지표 계산
acc = accuracy_score(test_true,test_pred)
prec = precision_score(test_true,test_pred,zero_division=0)
rec = recall_score(test_true,test_pred,zero_division=0)
f1 = f1_score(test_true,test_pred,zero_division=0)
auc = roc_auc_score(test_true,test_prob)
auroc = auc
print(f"Test Accuracy = {acc:.4f}")
print(f"Precision = {prec:.4f}")
print(f"Recall = {rec:.4f}")
print(f"F1 = {f1:.4f}")
print(f"AUC = {auc:.4f}")
print(f"AUROC = {auroc:.4f}")

# Confusion Matrix 시각화
cm = confusion_matrix(test_true,test_pred)
plt.figure(); plt.imshow(cm,interpolation='nearest'); plt.title('Confusion Matrix'); plt.colorbar()
for i in range(cm.shape[0]):
   for j in range(cm.shape[1]):
      plt.text(j,i,str(cm[i,j]),ha='center',va='center')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout(); plt.savefig('confusion_matrix.png'); plt.show()

# 결과 보고서 저장
with open('report_metrics.json','w',encoding='UTF-8') as open:
   json.dump({'Accuracy':acc,'Precision':prec,'Recall':rec,'F1':f1,'ROC-AUC':auc,'AUROC':auroc},open,indent=2,ensure_ascii=False)
print('평가 및 시각화 완료!')
print()

#Data 건수
print(f'Data 건수(Trainset) : {len(dtrain):,}건')
print(f'Data 건수(Testset) : {len(dtest):,}건')

#중복행
dup_train = dtrain.duplicated(subset=['title','content']).sum()
print(f'중복행(title + content) : {dup_train}건')

#Cosine Similarity 결과 상위 10개와 하위 10개 출력
print('[Cosine Similarity 상위 10개(제목과 내용이 매우 유사한 news)]')
print(dtrain[['title','content','cos_sim']].sort_values(by='cos_sim',ascending=False).head(10))
print('\n[Cosine Similarity 하위 10개(제목과 내용이 거의 다른 news)]')
print(dtrain[['title','content','cos_sim']].sort_values(by='cos_sim',ascending=True).head(10))

#전체 분포 시각화
plt.figure(figsize=(8,4))
plt.hist(dtrain['cos_sim'],bins=50,alpha=0.7,color='steelblue',edgecolor='black')
plt.title('Distribution of Cosine Similarity between Title and Content')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('cosine_similarity_distribution.png')
plt.show()

#Label별 평균 유사도 비교
avg_cos = dtrain.groupby('label')['cos_sim'].mean()
print('\n[label별 평균 Cosine Similarity]')
for label,mean in avg_cos.items():
    print(f'label = {label}→평균 유사도 = {mean:.4f}')