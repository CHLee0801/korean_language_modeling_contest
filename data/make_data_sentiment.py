import jsonlines
import pandas as pd
from random import sample
from sklearn.model_selection import train_test_split

positive_list = []
negative_list = []
neutral_list = []
file_1 = "data/modu_old_data.jsonl"
file_2 = "data/nikluge-sa-2022-train.jsonl"
file_3 = "data/nikluge-sa-2022-dev.jsonl"
sent_dict = {"positive":0,
             "negative":1,
             "neutral":2}
with jsonlines.open(file_1) as f:
    for line in f:
        sentence_input = line['sentence']
        if line["답"] == '긍정':
            continue
        elif line["답"] == '부정':
            negative_list.append([sentence_input, line['aspect'], 1])
        else:
            neutral_list.append([sentence_input, line['aspect'], 2])
            
with jsonlines.open(file_2) as f:
    for line in f:
        sentence_input = line['sentence_form']
        for a in line['annotation']:
            if a[2] == 'positive':
                positive_list.append([sentence_input, a[0], 0])
            if a[2] == 'negative':
                negative_list.append([sentence_input, a[0], 1])
            else:
                neutral_list.append([sentence_input, a[0], 2])

dev_list = []
with jsonlines.open(file_3) as f:
    for line in f:
        sentence_input = line['sentence_form']
        for a in line['annotation']:
            dev_list.append([sentence_input, a[0], sent_dict[a[2]]])
        

train_list = sample(positive_list, 804) + sample(negative_list, 400) + sample(neutral_list, 250)

headers = ["input", "entity", "label"]
output_df = pd.DataFrame(train_list)
output_df_2 = pd.DataFrame(dev_list)
output_df.to_csv("data/final_version/3_sentiment_train.csv", header=headers, index=False)
output_df_2.to_csv("data/final_version/3_sentiment_dev.csv", header=headers, index=False)

print("Sentiment completed!")