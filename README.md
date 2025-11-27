# Scientific Knowledge Question Answering | 과학 지식 질의 응답 시스템 구축
## Team

| <img width="150" height="150" src="https://github.com/user-attachments/assets/cef06034-aec1-40cb-a63a-367b5c341328"/> | <img width="150" height="150" src="https://github.com/user-attachments/assets/52d17b1d-3750-4986-88ce-a04eee9e18c2"/> | <img width="150" height="150" src="https://github.com/user-attachments/assets/2573c4a4-9cd2-4fa6-b419-a54b32dc823f"/> | <img width="150" height="150" src="https://github.com/user-attachments/assets/12f0ae79-f40e-4c13-b17b-469f2b3706a5"/> |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김시진](https://github.com/kimsijin33)             |            [임예슬](https://github.com/joy007fun)             |            [김상윤](https://github.com/94KSY)             |            [장윤정](https://github.com/yjjang06)             |
|                            IR 성능개선<br>프롬프트 엔지니어링                             |                            회의 진행<br>코드 개선                             |                            프롬프트 엔지니어링                             |                            프롬프트 엔지니어링                             |

## 0. Overview
### Environment
- Python 3.10+

### Requirements
```
sentence_transformers==2.2.2
elasticsearch==8.8.0
openai==1.7.2
```

## 1. Competiton Info

### Overview

- 질문과 이전 대화 히스토리를 보고 참고할 문서를 검색엔진에서 추출 후 이를 활용하여 질문에 적합한 대답을 생성하는 태스크입니다.

### Timeline

- November 14, 2025 - Start Date
- November 27, 2025 - Final submission deadline

## 2. Components

### Directory

```
├── code
│   ├── eda.ipynb
│   └── rag_with_elasticsearch.py
├── docs
│   └── Presentation.pdf
├── data
│   ├── eval.jsonl
│   └── documents.jsonl
```

## 3. Data descrption

### Dataset overview

- documents : 4,272개
- eval : 220개 (일반 대화 20개, 멀티턴 대화 20개 포함)

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- [pdf](https://github.com/AIBootcamp14/informationretrieval-ir_ad_1/blob/main/docs/Presentation.pdf)

## etc

### Meeting Log

- [Meeting Log Link](https://github.com/AIBootcamp14/informationretrieval-ir_ad_1/wiki/Meeting-Log)

### Reference

- [SentenceTransformer](https://pypi.org/project/sentence-transformers/)
- [Elasticsearch](https://elasticsearch-py.readthedocs.io/)
- [OpenAI](https://platform.openai.com/docs/api-reference/chat?lang=python)
- [Upstage](https://console.upstage.ai/docs/capabilities/generate/chat)
