# IP01069 - 인공지능과 정보보호

머신러닝 및 딥러닝 기법을 활용한 정보보호 응용을 학습하는 과목의 실습 저장소입니다.
Google Colab 기반의 주차별 실습 노트북과 기말 프로젝트를 포함합니다.

---

## 주차별 실습 내용

| 파일명 | 주제 | 핵심 알고리즘 / 기법 |
|--------|------|----------------------|
| `IP01069_2.ipynb` | K-최근접 이웃(KNN) 분류 | KNeighborsClassifier, 데이터 시각화, k값 튜닝 |
| `IP01069_3.ipynb` | 회귀 분석 및 규제 | KNN 회귀, 선형/다항 회귀, Ridge(L2), Lasso(L1) |
| `IP01069_5_1.ipynb` | 로지스틱 회귀 | 이진/다중 분류, Sigmoid, Softmax, predict_proba |
| `IP01069_5_2.ipynb` | 확률적 경사 하강법(SGD) | SGDClassifier, partial_fit, 에포크 튜닝, 학습 곡선 |
| `IP01069_6_1.ipynb` | 결정 트리 | DecisionTreeClassifier, 가지치기, 특성 중요도 |
| `IP01069_6_2.ipynb` | 교차 검증 및 하이퍼파라미터 튜닝 | K-Fold CV, GridSearchCV, RandomizedSearchCV |
| `IP01069_week9.ipynb` | 트리 앙상블 | Random Forest, Extra Trees, GBM, XGBoost, LightGBM |
| `IP01069_week10_2.ipynb` | K-Means 군집화 | KMeans, 엘보우 기법, 클러스터 시각화 |
| `IP01069_week10_3.ipynb` | 주성분 분석(PCA) | PCA, 분산 비율, 차원 축소 + KMeans 결합 |
| `IP01069_week12_1.ipynb` | 합성곱 신경망(CNN) | Conv2D, MaxPooling2D, Dropout, EarlyStopping |
| `IP01069_week12_2.ipynb` | CNN 시각화 및 Functional API | 필터/특성 맵 시각화, Keras Functional API |
| `IP01069_week12_3.ipynb` | CNN 심화 실습 | MNIST 적용, 모델 구조 비교 |
| `IP01069_week13_1.ipynb` | 순환 신경망(RNN) | Embedding, LSTM, GRU, Stacked RNN |
| `IP01069_week13_2.ipynb` | RNN 시퀀스 분류 | SimpleRNN, One-hot 인코딩, 시퀀스 패딩 |

---

## 기말 프로젝트 - 비트코인 거래 사기 탐지

| 파일 | 설명 |
|------|------|
| `기말프로젝트_2023111394_육은서.ipynb` | 프로젝트 코드 노트북 |
| `기말프로젝트보고서_2023111394_육은서.pdf` | 프로젝트 보고서 |

Elliptic Bitcoin Dataset을 활용하여 불법 거래를 탐지하는 프로젝트입니다.

- **데이터**: 46,564건의 비트코인 거래 (164개 특성)
- **전통 ML**: RandomForest + PCA — 정확도 97%, 정밀도 97%, 재현율 74%
- **그래프 특성 공학**: NetworkX 기반 In/Out-degree centrality, PageRank 추가 — 정확도 99%, 정밀도 100%, 재현율 87%
- **딥러닝(DNN)**: 256→128→64 은닉층 구조, Dropout(0.3), EarlyStopping 적용

---

## 사용 데이터셋

| 데이터셋 | 유형 | 활용 |
|----------|------|------|
| Fish-Market | 테이블 | KNN 분류/회귀 |
| Wine | 테이블 | 결정 트리, 앙상블, 교차 검증 |
| Iris | 테이블 | 로지스틱 회귀, 앙상블 |
| Fruits_300 | 이미지 (100x100) | K-Means 군집화, PCA |
| Fashion MNIST | 이미지 (28x28) | CNN |
| MNIST | 이미지 (28x28) | CNN |
| IMDB Reviews | 텍스트/시퀀스 | RNN, LSTM, GRU |
| Elliptic Bitcoin | 그래프/테이블 | 기말 프로젝트 (사기 탐지) |

---

## 기술 스택

- **ML**: scikit-learn, XGBoost, LightGBM
- **DL**: TensorFlow / Keras
- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **그래프 분석**: NetworkX
- **실행 환경**: Google Colab
