import os
import sys
import gc
import time
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List

# hugging face 모델 불러오기
from sentence_transformers import SentenceTransformer, util

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from datasets import Dataset

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
)

# seed 고정
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 설정
MODEL_NAME = "jhgan/ko-sroberta-multitask"
TRAIN_CSV = '../data/mission1_train.csv'
TEST_CSV = '../data/mission1_test.csv'
NUM_LABELS = 2  # 0=진짜, 1=가짜

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV,  encoding="euc-kr")

train_df.dropna(inplace=True)
train_df.reset_index(drop=True, inplace=True)

train_df["label"] = train_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

# dataset 형식 변환
train_dataset = Dataset.from_pandas(train_df[["title", "content", "label"]])
test_dataset = Dataset.from_pandas(test_df[["title", "content", "label"]])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(
        examples["title"],
        examples["content"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.config.problem_type = "single_label_classification"

training_args = TrainingArguments(
    report_to="none",               # 로그 툴 비활성화
    eval_strategy="epoch",          # epoch 단위 평가
    save_strategy="epoch",          # epoch 단위로 모델 저장
    learning_rate=2e-5,             # 학습률 ↓ (더 안정적인 수렴)
    per_device_train_batch_size=8,  # batch 크기 증가 (gradient 안정화)
    per_device_eval_batch_size=8,
    num_train_epochs=4,             # epoch 수 증가 (underfitting 방지)
    weight_decay=0.01,              # L2 정규화 유지
    warmup_ratio=0.1,               # learning rate warmup (초기 안정화)
    lr_scheduler_type="cosine",     # cosine scheduler로 smooth decay
    load_best_model_at_end=True,    # best model 로드
    metric_for_best_model="eval_auroc", # AUROC 기준으로 best model 선택
    greater_is_better=True,
    logging_steps=100,
    save_total_limit=2              # 체크포인트 저장 제한
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    try:
      auc = roc_auc_score(labels, probs)
    except ValueError:
      auc = float('nan')
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auroc": auc
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 학습
trainer.train()

SAVE_DIR = "./saved_sroberta_model_mission1"

trainer.save_model(SAVE_DIR)          # 모델 + config 저장
tokenizer.save_pretrained(SAVE_DIR)   # 토크나이저 저장

# SentenceTransformer
embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def compute_cosine_similarity(title_list, content_list):
  title_emb = embed_model.encode(title_list, convert_to_tensor=True, show_progress_bar=True)
  content_emb = embed_model.encode(content_list, convert_to_tensor=True, show_progress_bar=True)
  cosine_scores = util.cos_sim(title_emb, content_emb).diagonal()
  return cosine_scores.cpu().numpy()

train_df["cos_sim_sroberta"] = compute_cosine_similarity(train_df["title"], train_df["content"])
test_df["cos_sim_sroberta"] = compute_cosine_similarity(test_df["title"], test_df["content"])

pred_output = trainer.predict(tokenized_test)
logits, labels = pred_output.predictions, pred_output.label_ids
probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
preds = np.argmax(logits, axis=-1)

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="magma")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

train_metrics = trainer.state.log_history
train_acc, eval_acc, train_loss, eval_loss = [], [], [], []

for log in train_metrics:
    if 'eval_accuracy' in log:
        eval_acc.append(log['eval_accuracy'])
    if 'loss' in log and 'learning_rate' in log:
        train_loss.append(log['loss'])
    if 'eval_loss' in log:
        eval_loss.append(log['eval_loss'])

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(eval_acc, label='Validation Accuracy', color='orange')
plt.plot(np.linspace(0, len(eval_acc), len(train_loss)), 
         np.clip(train_loss, 0, 1), label='Training Accuracy (approx)', color='blue')
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_loss, label='Train Loss')
plt.plot(eval_loss, label='Eval Loss')
plt.title("Loss")
plt.legend()
plt.show()

fpr, tpr, _ = roc_curve(labels, probs)
auc = roc_auc_score(labels, probs)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

prec, rec, _ = precision_recall_curve(labels, probs)
plt.figure(figsize=(6,5))
plt.plot(rec, prec)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)
prec_val = precision_score(labels, preds)
rec_val = recall_score(labels, preds)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec_val:.4f}")
print(f"Recall: {rec_val:.4f}")
print(f"F1: {f1:.4f}")
print(f"AUROC: {auc:.4f}")

# 가짜뉴스(pred_label == 1)만 필터링
fake_df = test_df[test_df["pred_label"] == 1]

# 진짜뉴스만 필터링
real_df = test_df[test_df["pred_label"] == 0]

plt.figure(figsize=(8, 5))
sns.kdeplot(fake_df["cos_sim_sroberta"], color="red", label="Fake News (pred=1)")
sns.kdeplot(real_df["cos_sim_sroberta"], color="blue", label="Real News (pred=0)")
plt.title("Cosine Similarity Comparison: Fake vs Real News")
plt.xlabel("Cosine Similarity")
plt.legend()
plt.show()