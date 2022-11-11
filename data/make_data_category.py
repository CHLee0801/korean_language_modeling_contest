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

file_2 = "/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-train.jsonl"
file_3 = "/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-dev.jsonl"
sent_list = ["긍정", "부정", "중립", "없음"]
sent_dict = {
    "positive":"긍정",
    "negative":"부정",
    "neutral":"중립"
}

train_list = []
file_2_out = []
file_2_stats = [0,0,0,0]
first_list = []
second_list = []
third_list = []
fourth_list = []
with jsonlines.open(file_2) as f:
    for line in f:
        sentence_input = line['sentence_form']
        for a in line['annotation']:
            topic, category = a[0].split('#')
            sentiment = sent_dict[a[2]]
            what = [0,0,0,0,0,0,0]
            what[category_dict[category]] += 1
            query = f"{topic} 측면에 대한 감정은 {sentiment}이다."
            if [sentence_input, query, what] in train_list:
                continue
            train_list.append([sentence_input, query, what])

file_3_stats = [0,0,0,0]
dev_list = []
with jsonlines.open(file_3) as f:
    for line in f:
        sentence_input = line['sentence_form']
        for a in line['annotation']:
            topic, category = a[0].split('#')
            sentiment = sent_dict[a[2]]
            what = [0,0,0,0,0,0,0]
            what[category_dict[category]] += 1
            query = f"{topic} 측면에 대한 감정은 {sentiment}이다."
            if [sentence_input, query, what] in dev_list:
                continue
            dev_list.append([sentence_input, query, what])

print(len(train_list))
print(len(dev_list))

headers = ["input", "entity", "label"]
output_df = pd.DataFrame(train_list)
output_df_2 = pd.DataFrame(dev_list)
output_df.to_csv("/home/ubuntu/ch.lee/momal/data/ver4/category_train.csv", header=headers, index=False)
output_df_2.to_csv("/home/ubuntu/ch.lee/momal/data/ver4/category_dev.csv", header=headers, index=False)
exit()
train_list = []
file_2_out = []
file_2_stats = [0,0,0,0, 0, 0, 0]
first_list = []
second_list = []
third_list = []
fourth_list = []
with jsonlines.open(file_2) as f:
    for line in f:
        sentence_input = line['sentence_form']
        category_sub_list = []
        topic_sub_dict = {}
        for a in line['annotation']:
            topic, category = a[0].split('#')
            if topic not in topic_sub_dict:
                topic_sub_dict[topic] = [category_dict[category]]
            else:
                if category_dict[category] not in topic_sub_dict[topic]:
                    topic_sub_dict[topic].append(category_dict[category])
        
        for ttt in topic_sub_dict:
            what = [0,0,0,0, 0, 0, 0]
            for ww in topic_sub_dict[ttt]:
                what[ww] = 1
                file_2_stats[ww] += 1
            train_list.append([sentence_input, topic, what])


file_3_stats = [0,0,0,0, 0, 0, 0]
dev_list = []
with jsonlines.open(file_3) as f:
    for line in f:
        sentence_input = line['sentence_form']
        category_sub_list = []
        topic_sub_dict = {}
        for a in line['annotation']:
            topic, category = a[0].split('#')
            if topic not in topic_sub_dict:
                topic_sub_dict[topic] = [category_dict[category]]
            else:
                if category_dict[category] not in topic_sub_dict[topic]:
                    topic_sub_dict[topic].append(category_dict[category])
        
        for ttt in topic_sub_dict:
            what = [0,0,0,0, 0, 0, 0]
            for ww in topic_sub_dict[ttt]:
                what[ww] = 1
                file_3_stats[ww] += 1
            dev_list.append([sentence_input, topic, what])


print(len(train_list))
print(len(dev_list))
print(file_2_stats)
print(file_3_stats)
headers = ["input", "entity", "label"]
output_df = pd.DataFrame(train_list)
output_df_2 = pd.DataFrame(dev_list)
output_df.to_csv("/home/ubuntu/ch.lee/momal/data/ver2/category_train.csv", header=headers, index=False)
output_df_2.to_csv("/home/ubuntu/ch.lee/momal/data/ver2/category_dev.csv", header=headers, index=False)