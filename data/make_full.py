import jsonlines
import pandas as pd
from random import sample
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

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
topic_list = ['제품 전체', '본품', '패키지/구성품', '브랜드']
category_list = ['품질', '편의성', '일반', '다양성', '인지도', '가격', '디자인']
category_dict = {
    '품질' : 0,
    '편의성' : 1,
    '일반' : 2,
    '다양성' : 3,
    '인지도' : 4, 
    '가격' :5, 
    '디자인' : 6
}
category_dict_base = {
    '제품 전체' : ['품질', '편의성', '일반', '다양성', '인지도', '가격', '디자인'],
    '본품' : ['품질', '편의성', '일반', '다양성', '인지도', '가격', '디자인'], 
    '패키지/구성품' : ['품질', '편의성', '일반', '다양성', '가격', '디자인'], 
    '브랜드' : ['품질', '일반', '인지도', '가격', '디자인']
}
sent_list = ["긍정", "부정", "중립", "없음"]
sent_dict = {
    "positive":"긍정",
    "negative":"부정",
    "neutral":"중립"
}
file_2 = "/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-train.jsonl"
file_3 = "/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-dev.jsonl"

aspect_dict = {}
cnt = 0
for h in entity_property_pair:
    aspect_dict[h] = cnt
    cnt += 1

train_list = []
sent_dict = {
    "positive":"긍정",
    "negative":"부정",
    "neutral":"중립"
}
final = [0 for _ in range(len(entity_property_pair))]
final_2 = [0 for _ in range(len(entity_property_pair))]
with jsonlines.open(file_2) as f:
    for line in f:
        sent = line['sentence_form']
        what = [0 for _ in range(len(entity_property_pair))]
        for a in line['annotation']:
            what[aspect_dict[a[0]]] = 1
            final[aspect_dict[a[0]]] += 1
        train_list.append([sent, what])
     

dev_list = []
with jsonlines.open(file_3) as f:
    for line in f:
        sent = line['sentence_form']
        what = [0 for _ in range(len(entity_property_pair))]
        for a in line['annotation']:
            what[aspect_dict[a[0]]] = 1
            final_2[aspect_dict[a[0]]] += 1
        dev_list.append([sent, what])
      
print(final)
print(final_2)
exit()  
print(len(train_list))
print(len(dev_list))

headers = ["input", "label"]
output_df = pd.DataFrame(train_list)
output_df_2 = pd.DataFrame(dev_list)
output_df.to_csv("/home/ubuntu/ch.lee/momal/data/ver9/aspect_first_train.csv", header=headers, index=False)
output_df_2.to_csv("/home/ubuntu/ch.lee/momal/data/ver9/aspect_first_dev.csv", header=headers, index=False)

exit()
import jsonlines
import pandas as pd
from random import sample
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

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
topic_list = ['제품 전체', '본품', '패키지/구성품', '브랜드']
category_list = ['품질', '편의성', '일반', '다양성', '인지도', '가격', '디자인']
category_dict = {
    '품질' : 0,
    '편의성' : 1,
    '일반' : 2,
    '다양성' : 3,
    '인지도' : 4, 
    '가격' :5, 
    '디자인' : 6
}
category_dict_base = {
    '제품 전체' : ['품질', '편의성', '일반', '다양성', '인지도', '가격', '디자인'],
    '본품' : ['품질', '편의성', '일반', '다양성', '인지도', '가격', '디자인'], 
    '패키지/구성품' : ['품질', '편의성', '일반', '다양성', '가격', '디자인'], 
    '브랜드' : ['품질', '일반', '인지도', '가격', '디자인']
}
sent_list = ["긍정", "부정", "중립", "없음"]
sent_dict = {
    "positive":"긍정",
    "negative":"부정",
    "neutral":"중립"
}
file_2 = "/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-train.jsonl"
file_3 = "/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-dev.jsonl"

aspect_dict = {}
cnt = 0
for h in entity_property_pair:
    aspect_dict[h] = cnt
    cnt += 1

train_list = []
sent_dict = {
    "positive":"긍정",
    "negative":"부정",
    "neutral":"중립"
}
with jsonlines.open(file_2) as f:
    for line in f:
        sent = line['sentence_form']
        sub_dict = {}
        what = [0 for _ in range(len(entity_property_pair))]
        for a in line['annotation']:
            if a[2] not in sub_dict:
                sub_dict[a[2]] = [a[0]]
            else:
                sub_dict[a[2]].append(a[0])
        
        for w in sub_dict:        
            what = [0 for _ in range(len(entity_property_pair))]
            for ww in sub_dict[w]:
                what[aspect_dict[ww]] = 1
            train_list.append([sent, sent_dict[w], what])
     

dev_list = []
with jsonlines.open(file_3) as f:
    for line in f:
        sent = line['sentence_form']
        sub_dict = {}
        what = [0 for _ in range(len(entity_property_pair))]
        for a in line['annotation']:
            if a[2] not in sub_dict:
                sub_dict[a[2]] = [a[0]]
            else:
                sub_dict[a[2]].append(a[0])
        
        for w in sub_dict:        
            what = [0 for _ in range(len(entity_property_pair))]
            for ww in sub_dict[w]:
                what[aspect_dict[ww]] = 1
            dev_list.append([sent, sent_dict[w], what])
        
print(len(train_list))
print(len(dev_list))

headers = ["input", "entity", "label"]
output_df = pd.DataFrame(train_list)
output_df_2 = pd.DataFrame(dev_list)
output_df.to_csv("/home/ubuntu/ch.lee/momal/data/ver9/aspect_second_train.csv", header=headers, index=False)
output_df_2.to_csv("/home/ubuntu/ch.lee/momal/data/ver9/aspect_second_dev.csv", header=headers, index=False)