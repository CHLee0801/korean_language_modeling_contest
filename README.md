# 2022 국립국어원 인공 지능 언어 능력 평가 본선 모델

2022 국립국어원 상위 10개팀에 선정된 모델입니다.

## 0. 대회 설명 및 프로젝트 설명

본 대회는 상품이나 서비스에 대한 리뷰가 주어졌을 때 Aspect와 그에 상응하는 Sentiment를 구분하는 ABSA Task를 수행하는 대회입니다.

Aspect로는     
```
'제품 전체#품질', '제품 전체#편의성', '제품 전체#일반', '제품 전체#다양성', '제품 전체#인지도', '제품 전체#가격', '제품 전체#디자인',
'본품#품질', '본품#편의성', '본품#일반', '본품#다양성', '본품#인지도', '본품#가격', '본품#디자인',
'패키지/구성품#품질', '패키지/구성품#편의성', '패키지/구성품#일반', '패키지/구성품#다양성', '패키지/구성품#가격', '패키지/구성품#디자인', 
'브랜드#품질', '브랜드#일반', '브랜드#인지도', '브랜드#가격', '브랜드#디자인'
```
가 있고, Sentiment는 '긍정', '부정', '중립' 이렇게 3가지 입니다.

### Baseline Model

Baseline Model은 'https://github.com/kiyoungkim1/LMkor' 링크에서 제공하는 KoBert를 사용하였습니다.

해당 모델이 사용한 학습데이터는 다음과 같고, '모두의 말뭉치'를 학습하였기 때문에 Baseline model로 선정하였습니다.
```
- 국내 주요 커머스 리뷰 1억개 + 블로그 형 웹사이트 2000만개 (75GB)
- 모두의 말뭉치 (18GB)
- 위키피디아와 나무위키 (6GB)
```

### 문제 해결 방법

데이터를 분석하였을 때 Aspect를 '#'을 기준으로 나누어 먼저 Aspect를 구하고 Sentiment를 구하는 방식이 가장 효율적이라고 판단하였습니다.

Aspect는 'Topic#Category' 로 구성되어 있다고 판단했고,

Topic은 [제품 전체, 본품, 패키지/구성품, 브랜드] 이렇게 4-way로 나뉘고,

Category는 [품질, 일반, 편의성, 디자인, 인지도, 가격, 다양성] 이렇게 7-way로 나뉩니다.

데이터 분석 결과 Category에서 Label Imbalance가 있었지만 [품질, 일반, 나머지]로 묶었을 때 규형된 Label Distribution이 있었습니다.

따라서,

1. 전체 데이터에 대해 Category를 3-way로 분류,
2. 나머지로 분류된 데이터들의 Category를 5-way로 분류,
3. Category를 Prompt로 주고 Topic을 4-way로 분류,
4. Aspect(Topic + Category)를 주고 Sentiment를 3-way로 분류

이러한 방식으로 모델을 생성했습니다. 모든 Aspect에 대해서 Label Imbalance가 존재했고, Dev set에 Overfitting 되는 문제를 방지하기 위해 Cross-Validation을 해서 5개의 모델을 구현하고, 5개의 모델 결과를 앙상블하여 최종 Submission File을 생성하였습니다.

## 1. 콘다 환경 설정 및 필요한 라이브러리 설치
```
conda create -n momal python=3.9 && conda activate momal
pip install -r requirements.txt
```
CUDA 버전과 개발 환경에 맞게 Pytorch를 설치해주세요.
https://pytorch.org/ 를 참고하세요.
```
#For CUDA 10.x
pip3 install torch torchvision torchaudio
#For CUDA 11.x
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## 2. 데이터 제작 및 모델 Checkpoints 다운로드
데이터를 만드는 코드입니다. 총 4가지 스테이지에 맞게 Train, Dev 파일이 생성됩니다.
```
bash make_data.sh
```

Cross Validation에서 가장 높은 성능을 보인 5개의 모델 Checkpoint입니다. Section 3, Step 4에서 바로 활용해 최종 파일 제작을 할 수 있습니다.
```
gdown https://drive.google.com/uc?id=12DWUApFf25YC_BnKRyRjxOqbeZeKUilK
unzip ckpt.zip
```

## 3. 실험 실행 코드 (다운받은 Checkpoint를 활용해 바로 데이터를 만들고자 한다면 Step 4로 가세요)

각 실험의 최고 성능을 알기 위해서는 wandb 회원 가입 후 로그인을 하여야합니다.

최고 성능을 가진 checkpoint epoch를 확인할 수 있고, 그 모델들을 이용해 최종 모델을 제작합니다.

### Step 1-1 -> 카테고리 3-way 분류 모델 실행 코드 (품질, 일반, 나머지)

```
python run.py --config configs/bert/bert_category_3way.json
```

### Step 1-2 -> 카테고리  5-way 분류 모델 실행 코드 (편의성, 디자인, 인지도, 가격, 다양성)

```
python run.py --config configs/bert/bert_category_5way.json
```

### Step 2 -> 토픽 4-way 분류 모델 실행 코드 (제품 전체, 본품, 패키지/구성품, 브랜드) 

```
python run.py --config configs/bert/bert_topic.json
```

### Step 3 -> 감정 3-way 분류 모델 실행 코드 (긍정, 부정, 중립)

```
python run.py --config configs/bert/bert_sentiment.json
```

### Step 4 -> 각 모델에서 최고 Checkpoints 경로를 기입하여 5개의 Output_file 생성

config 폴더에 있는 'write_*.json' 파일에서 'checkpoint_path_*'를 수정해주어야 합니다.

> checkpoint_path_1 : Step 1-1의 모델 Checkpoints

> checkpoint_path_2 : Step 1-2의 모델 Checkpoints

> checkpoint_path_3 : Step 2의 모델 Checkpoints

> checkpoint_path_4 : Step 3의 모델 Checkpoints

Step 2에서 Checkpoint를 받아서 활용할 경우 config 파일들을 수정할 필요가 없이 바로 Output_file 생성이 가능합니다.

아래 명령어를 통해 5개의 모델 output_file을 생성합니다.

```
python run.py --config configs/write_0.json
python run.py --config configs/write_1.json
python run.py --config configs/write_2.json
python run.py --config configs/write_3.json
python run.py --config configs/write_4.json
```

output_file 폴더 아래 5개 모델에 대한 output file이 생성됩니다.

### Step 5 -> 5개의 모델 결과를 활용해 최종 Submission File 생성

blahblahblahblahblahblahblahblahblahblah