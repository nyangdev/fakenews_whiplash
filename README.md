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

## 아쉬웠던 점
이번 프로젝트를 진행하면서 아쉬웠던 부분은 다음과 같습니다.
1) SRoBERTa만 한국어 사전학습 모델을 사용했다는 점
2) 단어 기반 임베딩 모델과 문장 기반 임베딩 모델의 비교 자체가 본질적으로 조건이 다르다는 점
3) 시간 부족으로 논문에서 제공하는 데이터를 동일하게 사용했다는 점과 그로 인한 데이터 부족

## 향후 개선 계획
1) Word2Vec, FastText 코드 리팩토링 예정
2) Word2Vec과 FastText를 한국어 학습 진행 후 세 모델을 다시 비교 예정
3) 뉴스 기사를 직접 크롤링 하는 작업을 통해 데이터를 늘릴 예정
4) 문장 기반 모델(KoELECTRA, KoBERT 등) 들을 모아서 SRoBERTa와 비교해볼 예정