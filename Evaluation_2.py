from re import M
from Datasets import Custom_Dataset
from torch.utils.data import DataLoader
import os
import torch
import jsonlines
import csv
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch import nn
import pandas as pd
import numpy as np
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list

def evaluation_f1(true_data, pred_data):
    
    true_data_list = true_data
    pred_data_list = pred_data

    ce_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    pipeline_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    for i in range(len(true_data_list)):

        # TP, FN checking
        is_ce_found = False
        is_pipeline_found = False
        for y_ano  in true_data_list[i]['annotation']:
            y_category = y_ano[0]
            y_polarity = y_ano[2]

            for p_ano in pred_data_list[i]['annotation']:
                p_category = p_ano[0]
                p_polarity = p_ano[1]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True

                    break

            if is_ce_found is True:
                ce_eval['TP'] += 1
            else:
                ce_eval['FN'] += 1

            if is_pipeline_found is True:
                pipeline_eval['TP'] += 1
            else:
                pipeline_eval['FN'] += 1

            is_ce_found = False
            is_pipeline_found = False

        # FP checking
        for p_ano in pred_data_list[i]['annotation']:
            p_category = p_ano[0]
            p_polarity = p_ano[1]

            for y_ano  in true_data_list[i]['annotation']:
                y_category = y_ano[0]
                y_polarity = y_ano[2]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True

                    break

            if is_ce_found is False:
                ce_eval['FP'] += 1

            if is_pipeline_found is False:
                pipeline_eval['FP'] += 1

    ce_precision = ce_eval['TP']/(ce_eval['TP']+ce_eval['FP'])
    ce_recall = ce_eval['TP']/(ce_eval['TP']+ce_eval['FN'])

    ce_result = {
        'Precision': ce_precision,
        'Recall': ce_recall,
        'F1': 2*ce_recall*ce_precision/(ce_recall+ce_precision)
    }

    pipeline_precision = pipeline_eval['TP']/(pipeline_eval['TP']+pipeline_eval['FP'])
    pipeline_recall = pipeline_eval['TP']/(pipeline_eval['TP']+pipeline_eval['FN'])

    pipeline_result = {
        'Precision': pipeline_precision,
        'Recall': pipeline_recall,
        'F1': 2*pipeline_recall*pipeline_precision/(pipeline_recall+pipeline_precision)
    }

    return {
        'category extraction result': ce_result,
        'entire pipeline result': pipeline_result
    }

def evaluate(args, Model_1, Model_2):

    topic_list = ['제품 전체', '본품', '패키지/구성품', '브랜드']
    category_list = ['품질', '편의성', '일반', '다양성', '인지도', '가격', '디자인']
    sent_list = ['긍정', '부정', '중립']
    sent_dict = {
        '긍정' : "positive",
        '부정' : "negative",
        '중립' : "netural"
    }

    ############### model_1 ###############
    model_1 = Model_1(args, "binary")
    mode_1, mode_2 = args.mode.split(', ')
    topic_grad = {}
    topic_classifier_grad = {}
    if args.checkpoint_path_1 != "":
        sent_model = torch.load(args.checkpoint_path_1)
        
        for key, value in sent_model.items():
            if 'classifier' in key:
                topic_classifier_grad[key.split('.')[-1]] = value
            else:
                topic_grad['model.'+key] = value
        model_1.load_state_dict(topic_grad, strict=False)
    
    model_1.eval()
    model_1.to('cuda')
    tokenizer_1 = model_1.tokenizer
    topic_classifier = model_1.labels_classifier
    topic_classifier.load_state_dict(topic_classifier_grad)

    ############### model_2 ###############

    model_2 = Model_1(args, "category")
    
    category_grad={}
    category_classifier_grad = {}
    if args.checkpoint_path_2 != "":
        cat_model = torch.load(args.checkpoint_path_2)
        
        for key, value in cat_model.items():
            if 'classifier' in key:
                category_classifier_grad[key.split('.')[-1]] = value
            else:
                category_grad['model.'+key] = value
        model_2.load_state_dict(category_grad, strict=False)
    
    model_2.eval()
    model_2.to('cuda')
    tokenizer_2 = model_2.tokenizer
    category_classifier = model_2.labels_classifier
    category_classifier.load_state_dict(category_classifier_grad)

    #########################################################

    # If folder doesn't exist, then create it.
    MYDIR = ("/".join((args.output_log.split('/'))[:-1]))
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")

    if args.test == True:
        true_data = jsonlload("/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-test.jsonl")
        sentence_list_for_topic = []
        for tdata in true_data:
            for top in topic_list:
                for sent in sent_list:
                    if mode_1 == 'nli':
                        query = f"{sent} - {top}"
                    else:
                        query = f"{top} 측면에 대한 감정은 {sent}이다."
                    sentence_list_for_topic.append([tdata['sentence_form'], query])
        dataset_1 = Custom_Dataset(sentence_list_for_topic, "eval", "binary", tokenizer_1, args.input_length)
    else:
        dataset_1 = Custom_Dataset(args.eval_path, "valid", "binary", tokenizer_1, args.input_length)
    
    print('Length of category validation data: ',len(dataset_1))
    loader_1 = DataLoader(dataset_1, batch_size=args.eval_batch_size, shuffle=False)

    topic_pred = []
    topic_gt = []
    topic_temp = np.array([])

    for batch in tqdm(iter(loader_1)):
        with torch.no_grad():
            output = model_1.model(
                batch['source_ids'].cuda(),
                batch['source_mask'].cuda()
            )
            output = topic_classifier(output.pooler_output)
            output = torch.sigmoid(output)

            if args.test == False:
                topic_gt.append(batch['labels'][0].detach().cpu())
            topic_pred.append(int(torch.argmax(output).detach().cpu()))
            topic_temp = np.append(topic_temp, np.array([output[0][1].detach().cpu()]))

    first_out = []

    if args.test == False:
        first_dataset = pd.read_csv(args.eval_path, encoding='utf-8')
        for idx, row in first_dataset.iterrows():
            if idx % 12 == 0 and sum(topic_pred[idx:idx+12]) == 0:
                sub_sub_list = topic_pred[idx:idx+12]
                max_index = np.argmax(sub_sub_list)
                topic = topic_list[max_index//4]
                sentiment = sent_list[max_index%3]
                if mode_2 == 'nli':
                    query = f"{sentiment} - {topic}"
                else:
                    query = f'{topic} 측면에 대한 감정은 {sentiment}이다.'
                first_out.append([row['input'], query])
            if topic_pred[idx] == 1:
                entity = row['entity']
                if mode_1 == 'nli':
                    sentiment, topic = entity.split(' - ')
                else:
                    sentiment, topic = entity.split('감정은 ')[-1][:2], entity.split(' 측면에')[0]
                if sentiment == '없음':
                    continue
                if mode_2 == 'nli':
                    query = f"{sentiment} - {topic}"
                else:
                    query = f'{topic} 측면에 대한 감정은 {sentiment}이다.'
                first_out.append([row['input'], query])
    else:
        for idx in range(len(sentence_list_for_topic)):
            if idx % 12 == 0 and sum(topic_pred[idx:idx+12]) == 0:
                sub_sub_list = topic_pred[idx:idx+12]
                max_index = np.argmax(topic_temp)
                topic = topic_list[max_index//4]
                sentiment = sent_list[max_index%3]
                if mode_2 == 'nli':
                    query = f"{sentiment} - {topic}"
                else:
                    query = f'{topic} 측면에 대한 감정은 {sentiment}이다.'
                first_out.append([row['input'], query])
                
            if topic_pred[idx] == 1:
                entity = sentence_list_for_topic[idx][1]
                if mode_1 == 'nli':
                    sentiment, topic = entity.split(' - ')
                else:
                    sentiment, topic = entity.split('감정은 ')[-1][:2], entity.split(' 측면에')[0]
                if sentiment == '없음':
                    continue
                if mode_2 == 'nli':
                    query = f"{sentiment} - {topic}"
                else:
                    query = f'{topic} 측면에 대한 감정은 {sentiment}이다.'
                first_out.append([sentence_list_for_topic[idx][0], query])

    dataset_2 = Custom_Dataset(first_out, "eval", "category", tokenizer_2, args.input_length)
    print('Length of topic validation data: ',len(dataset_2))
    loader_2 = DataLoader(dataset_2, batch_size=args.eval_batch_size, shuffle=False)

    category_pred = []
    for batch in tqdm(iter(loader_2)):
        with torch.no_grad():
            output = model_2.model(
                batch['source_ids'].cuda(),
                batch['source_mask'].cuda()
            )
            output = category_classifier(output.pooler_output)
            output = torch.sigmoid(output)

            category_pred.append(int(torch.argmax(output).detach().cpu()))
    
    final_output_dict = {}
    for idd in range(len(category_pred)):
        entity = first_out[idd][1]
        if mode_2 == 'nli':
            sentiment, topic = entity.split(' - ')
        else:
            sentiment, topic = entity.split('감정은 ')[-1][:2], entity.split(' 측면에')[0]
        aspect = f'{topic}#{category_list[category_pred[idd]]}'
        sentiment = sent_dict[sentiment]
        if first_out[idd][0] not in final_output_dict:
            final_output_dict[first_out[idd][0]] = [[aspect, sentiment]]
        else:
            final_output_dict[first_out[idd][0]].append([aspect, sentiment])

    if args.test == False:
        true_data = jsonlload("/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-dev.jsonl")
        pred_data = []
        for data in true_data:
            annotation_list = []
            if data['sentence_form'] in final_output_dict:
                annotation_list = final_output_dict[data['sentence_form']]
            sample_dict = {
                'id':data['id'],
                'sentence_form':data['sentence_form'],
                'annotation':annotation_list
            }
            pred_data.append(sample_dict)
        
        print(evaluation_f1(true_data, pred_data))
    else:
        true_data = jsonlload("/home/ubuntu/ch.lee/momal/data/nikluge-sa-2022-test.jsonl")
        pred_data = []
        for data in true_data:
            annotation_list = []
            if data['sentence_form'] in final_output_dict:
                annotation_list = final_output_dict[data['sentence_form']]
            sample_dict = {
                'id':data['id'],
                'sentence_form':data['sentence_form'],
                'annotation':annotation_list
            }
            pred_data.append(sample_dict)
        outfile_name = "/home/ubuntu/ch.lee/momal/output_file/results.jsonl"
        with open(outfile_name , encoding= "utf-8" ,mode="w") as file: 
            for i in pred_data: 
                file.write(json.dumps(i,ensure_ascii=False) + "\n")
    