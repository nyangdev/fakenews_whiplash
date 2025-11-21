# 뉴스 데이터 학습을 통한 가짜 뉴스 분류
25.11.03 ~ 25.11.14

**Whiplash**

팀장: 김민지

팀원: 이창주, 박주언

## 프로젝트 구조
```
fakenews
├── README.md
├── data
│   ├── mission1_test.csv // 임무 1 테스트 데이터
│   ├── mission1_train.csv // 임무 1 훈련 데이터
│   ├── mission2_test.csv // 임무 2 테스트 데이터
│   └── mission2_train.csv // 임무 2 훈련 데이터
├── docs
│   ├── fakenews_발표자료.pdf
│   ├── 분석 결과 보고서(가짜 news 탐지자동화).pdf
│   └── 분석 계획 보고서(가짜 news 탐지 자동화).pdf
└── model
    ├── whiplash_FastText_mission1.py
    ├── whiplash_FastText_mission2.py
    ├── whiplash_SRoBERTa_mission1.py
    ├── whiplash_SRoBERTa_mission2.py
    ├── whiplash_Word2vec_mission1.py
    └── whiplash_Word2vec_mission2.py

```

## 프로젝트 개요
본 프로젝트는 한국어 뉴스 기사에서 발생하는 가짜뉴스를 탐지하기 위해 NLP 모델들을 실제 데이터에 적용해보고, 각 모델의 성능을 비교하는 실험 연구입니다.

### 가짜뉴스 유형
가짜뉴스 유형은 두 가지로 정의합니다.
1) 임무1 - **제목 본문 부정합**
- 기사 제목과 본문 내용이 서로 일치하지 않는 경우

2) 임무 2 - **본문 내 문맥 불일치**
- 본문 일부 문장이 전체 문맥과 관련이 없는 경우

### 사용된 모델
사용한 모델은 3가지로, 다음과 같습니다.
1) **Word2Vec**
2) **FastText**
3) **SRoBERTa** (HuggingFace에서 제공하는 ko-SRoBERTa 사용)

### 사용된 데이터셋
[이동호, 이정훈, 김유리, 정윤철, 박승민, 양유정, 신웅비, “딥러닝 기법을 이용한 가짜뉴스 탐지,” 2018년 춘계학술발표대회 논문집, 제52권 제1호, 2018.]

논문에서 제공한 데이터셋을 사용했습니다. (추후 보강하여 재학습 예정)

### 선정한 평가지표
- Accuracy
- AUROC

## 팀 역할
이번 프로젝트에서는 모델별로 담당자를 나누어 진행했습니다.
- SRoBERT 모델 - 김민지
    - HuggingFace에서 제공하는 ko-SRoBERTa 기반으로 Mission1 / Mission2 실험 수행
    - 분석 계획서, 분석 결과 보고서 작성

- FastText 모델 - 이창주
    - FastText 임베딩 기반 분류 모델 구현
    - Mission1 / Mission2 실험 수행

- Word2Vec 모델 - 박주언
    - Word2Vec 임베딩 기반 분류 모델 구현
    - Mission1 / Mission2 실험 수행
    - 발표 자료 제작
