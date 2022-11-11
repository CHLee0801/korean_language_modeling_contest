import jsonlines
import pandas as pd
from random import sample
from sklearn.model_selection import train_test_split

headers = ["input", "entity", "label"]
final_list = []
positive_list = []
negative_list = []
neutral_list = []
file_1 = "/home/ubuntu/ch.lee/momal/original_data/modu_aspect_sentiment/instructions_modu_aspect_sentiment.jsonl"
file_2 = "/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-train.jsonl"
file_3 = "/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-dev.jsonl"
sent_dict = {"positive":0,
             "negative":1,
             "neutral":2}
with jsonlines.open(file_1) as f:
    for line in f:
        sentence_input = line['sentence']
        if line["answer"] == '긍정':
            continue
            positive_list.append([sentence_input, line["aspect"], 0])
        elif line["answer"] == '부정':
            final_list.append([sentence_input, [0,1,0]])
        else:
            final_list.append([sentence_input, [0,0,1]])
            
with jsonlines.open(file_2) as f:
    for line in f:
        sentence_input = line['sentence_form']
        what = [0,0,0]
        for a in line['annotation']:
            what[sent_dict[a[2]]] = 1
        if sum(what) > 1 or what == [0,0,1] or what == [0,1,0]:
            final_list.append([sentence_input, what])
        else:
            positive_list.append([sentence_input, what])
dev_list = []
with jsonlines.open(file_3) as f:
    for line in f:
        sentence_input = line['sentence_form']
        what = [0,0,0]
        for a in line['annotation']:
            what[sent_dict[a[2]]] = 1
        dev_list.append([sentence_input, what])
        
print(len(positive_list))


final_list = sample(positive_list, 600) + final_list
print(len(final_list))
headers = ["input", "label"]
output_df = pd.DataFrame(final_list)
output_df_2 = pd.DataFrame(dev_list)
output_df.to_csv("/home/ubuntu/ch.lee/momal/data/ver7/sentiment_train_3.csv", header=headers, index=False)
#output_df_2.to_csv("/home/ubuntu/ch.lee/momal/data/ver7/sentiment_dev.csv", header=headers, index=False)