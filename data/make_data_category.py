import jsonlines
import pandas as pd
from random import sample
from sklearn.model_selection import train_test_split

entity_property_pair = [
    '제품 전체#품질', '제품 전체#편의성', '제품 전체#일반', '제품 전체#다양성', '제품 전체#인지도', '제품 전체#가격', '제품 전체#디자인',
    '본품#품질', '본품#편의성', '본품#일반', '본품#다양성', '본품#인지도', '본품#가격', '본품#디자인',
    '패키지/구성품#품질', '패키지/구성품#편의성', '패키지/구성품#일반', '패키지/구성품#다양성', '패키지/구성품#가격', '패키지/구성품#디자인', 
    '브랜드#품질', '브랜드#일반', '브랜드#인지도', '브랜드#가격', '브랜드#디자인'
]

topic_dict = {
    '제품 전체' : 0, 
    '본품' : 1, 
    '패키지/구성품' : 2, 
    '브랜드' : 3
}
category_dict = {
    '품질' : 0,
    '일반' : 1,
    '편의성' : 2,
    '디자인' : 3,
    '인지도' : 4, 
    '가격' :5, 
    '다양성' : 6
}

category_dict_base = {
    '제품 전체' : ['품질', '편의성', '일반', '다양성', '인지도', '가격', '디자인'],
    '본품' : ['품질', '편의성', '일반', '다양성', '인지도', '가격', '디자인'], 
    '패키지/구성품' : ['품질', '편의성', '일반', '다양성', '가격', '디자인'], 
    '브랜드' : ['품질', '일반', '인지도', '가격', '디자인']
}

file_2 = "data/nikluge-sa-2022-train.jsonl"
file_3 = "data/nikluge-sa-2022-dev.jsonl"
sent_list = ["긍정", "부정", "중립"]
sent_dict = {
    "positive":"긍정",
    "negative":"부정",
    "neutral":"중립"
}

train_list_3way = []
train_list_5way = []

with jsonlines.open(file_2) as f:
    for line in f:
        sentence_input = line['sentence_form']
        category_sub_dict = {}
        for a in line['annotation']:
            topic, category = a[0].split('#')
            sentiment = sent_dict[a[2]]
            category_sub_dict[category] = category_dict[category]

        category_3way_list = [0,0,0]
        category_5way_list = [0,0,0,0,0]
        for cat in category_sub_dict:
            if category_sub_dict[cat] < 2:
                category_3way_list[category_sub_dict[cat]] = 1
            else:
                category_3way_list[2] = 1
                category_5way_list[category_sub_dict[cat]-2] = 1

        if category_3way_list != [0,0,0]:
            train_list_3way.append([sentence_input, category_3way_list])
        if category_5way_list != [0,0,0,0,0]:
            train_list_5way.append([sentence_input, category_5way_list])


dev_list_3way = []
dev_list_5way = []

with jsonlines.open(file_3) as f:
    for line in f:
        sentence_input = line['sentence_form']
        category_sub_dict = {}
        for a in line['annotation']:
            topic, category = a[0].split('#')
            sentiment = sent_dict[a[2]]
            category_sub_dict[category] = category_dict[category]

        category_3way_list = [0,0,0]
        category_5way_list = [0,0,0,0,0]
        for cat in category_sub_dict:
            if category_sub_dict[cat] < 2:
                category_3way_list[category_sub_dict[cat]] = 1
            else:
                category_3way_list[2] = 1
                category_5way_list[category_sub_dict[cat]-2] = 1

        if category_3way_list != [0,0,0]:
            dev_list_3way.append([sentence_input, category_3way_list])
        if category_5way_list != [0,0,0,0,0]:
            dev_list_5way.append([sentence_input, category_5way_list])

headers = ["input", "label"]
output_df_0 = pd.DataFrame(train_list_3way)
output_df_1 = pd.DataFrame(dev_list_3way)
output_df_2 = pd.DataFrame(train_list_5way)
output_df_3 = pd.DataFrame(dev_list_5way)
output_df_0.to_csv("data/final_version/0_category_3way_train.csv", header=headers, index=False)
output_df_1.to_csv("data/final_version/0_category_3way_dev.csv", header=headers, index=False)
output_df_2.to_csv("data/final_version/1_category_5way_train.csv", header=headers, index=False)
output_df_3.to_csv("data/final_version/1_category_5way_dev.csv", header=headers, index=False)

print("Category completed!")