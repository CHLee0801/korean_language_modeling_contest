# 2022 국립국어원 인공 지능 언어 능력 평가 본선 모델

In order to reproduce our results, take the following steps:
## 1. Create conda environment and install requirements
```
conda create -n momal python=3.9 && conda activate momal
pip install -r requirements.txt
```

Also, make sure to install the correct version of pytorch corresponding to the CUDA version and environment:
Refer to https://pytorch.org/
```
#For CUDA 10.x
pip3 install torch torchvision torchaudio
#For CUDA 11.x
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## 2. Reproduce train data and Download ckpt 
To reproduce data needed for this project:
```
bash make_data.sh
```

To download Final Checkpoints and unzip it:
```
gdown https://drive.google.com/uc?id=12DWUApFf25YC_BnKRyRjxOqbeZeKUilK
unzip ckpt.zip
```

## 3. Run the experiment and generate output file.

To see the results, you have to login to wandb. You will be able to choose best checkpoints out of all the experiments.

### Step 1-1 -> Classify category in 3-way (품질, 일반, 나머지)

```
python run.py --config configs/bert/bert_category_3way.json
```

### Step 1-2 -> Classify category in 5-way (편의성, 디자인, 인지도, 가격, 다양성)

```
python run.py --config configs/bert/bert_category_5way.json
```

### Step 2 -> Classify topic in 4-way (제품 전체, 본품, 패키지/구성품, 브랜드) 

```
python run.py --config configs/bert/bert_topic.json
```

### Step 3 -> Classify sentiment in 3-way (긍정, 부정, 중립)

```
python run.py --config configs/bert/bert_sentiment.json
```

### Step 4 -> Generate outfile per each fold by using best checkpoints.

For this part, you have to fill in "checkpoint_path_#" in config file. 

For convenience, we provide checkpoints in Section 2. If you downloaded the ckpt, you just run the following command, or you have to modify each file listed below.

```
python run.py --config configs/write_0.json
python run.py --config configs/write_1.json
python run.py --config configs/write_2.json
python run.py --config configs/write_3.json
python run.py --config configs/write_4.json
```

### Step 5 -> Generate final submission file using voting strategy.

blahblahblahblahblahblahblahblahblahblah