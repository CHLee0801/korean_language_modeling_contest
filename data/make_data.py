

##### Aspect main

import jsonlines
import pandas as pd
import json
import random

file_1 = "/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-train.jsonl"
file_2 = "/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-dev.jsonl"

topic_list = ['제품 전체', '본품', '브랜드', '패키지/구성품']
category_dict = {
    '제품 전체':["품질", "편의성", "일반", "다양성", "인지도", "가격", "디자인"],
    "본품":["품질", "편의성", "일반", "다양성", "인지도", "가격", "디자인"],
    "패키지/구성품":["품질", "편의성", "일반", "다양성", "가격", "디자인"],
    "브랜드":["품질", "일반", "인지도", "가격", "디자인"]
}

final_list = []

gather_up_dict = {}

with jsonlines.open(file_1) as f:
    cnt = 0
    for line in f:
        if '~' in line['annotation']:
            cnt += 1
            sentiment_list = []
            for a in line['annotation']:
                if a[2] not in sentiment_list:
                    sentiment_list.append(a[2])
            if "negative" in sentiment_list or "neutral" in sentiment_list:
                print(line)

print(cnt)

with jsonlines.open(file_2) as f:
    cnt = 0
    for line in f:
        if '~' in line['annotation']:
            cnt += 1
            sentiment_list = []
            for a in line['annotation']:
                if a[2] not in sentiment_list:
                    sentiment_list.append(a[2])
            if "negative" in sentiment_list or "positive" in sentiment_list:
                print(line)

print(cnt)