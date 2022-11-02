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

def evaluate(args, Model_1, Model_2, Model_3):

    topic_list = ['제품 전체', '본품', '패키지/구성품', '브랜드']
    category_list = ['편의성', '디자인', '인지도','가격','다양성']

    dropout = nn.Dropout(p=0.1)
    ############### model_0 ###############
    
    model_0 = Model_1(args, "trinary")

    topic_grad = {}
    topic_classifier_grad = {}
    if args.checkpoint_path_1 != "":
        sent_model = torch.load(args.checkpoint_path_1)
        
        for key, value in sent_model.items():
            if 'classifier' in key:
                topic_classifier_grad[key.split('.')[-1]] = value
            else:
                topic_grad['model.'+key] = value

        model_0.load_state_dict(topic_grad, strict=False)
    
    model_0.eval()
    model_0.to('cuda')
    tokenizer_0 = model_0.tokenizer
    first_classifier = model_0.labels_classifier
    first_classifier.load_state_dict(topic_classifier_grad)
    
    ############### model_1 ###############

    model_1 = Model_1(args, "topic")

    topic_grad = {}
    topic_classifier_grad = {}
    if args.checkpoint_path_2 != "":
        sent_model = torch.load(args.checkpoint_path_2)
        
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
    if args.checkpoint_path_3 != "":
        cat_model = torch.load(args.checkpoint_path_3)
        
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

    ############### model_3 ###############

    model_3 = Model_2(args, "sentiment")
    
    sentiment_grad={}
    sent_classifier_grad = {}
    if args.checkpoint_path_4 != "":
        sent_model = torch.load(args.checkpoint_path_4)
        
        for key, value in sent_model.items():
            if 'classifier' in key:
                sent_classifier_grad[key.split('.')[-1]] = value
            else:
                sentiment_grad['model.'+key] = value
        model_3.load_state_dict(sentiment_grad, strict=False)
    
    model_3.eval()
    model_3.to('cuda')
    tokenizer_3 = model_3.tokenizer
    sentiment_classifier = model_3.labels_classifier
    sentiment_classifier.load_state_dict(sent_classifier_grad)

    #########################################################
    """
    ############### model_4 ###############

    model_4 = Model_4(args)
    
    sentiment_grad_2={}
    sent_classifier_grad_2 = {}
    if args.checkpoint_path_4 != "":
        sent_model_2 = torch.load(args.checkpoint_path_4)
        
        for key, value in sent_model_2.items():
            if 'classifier' in key:
                sent_classifier_grad_2[key.split('.')[-1]] = value
            else:
                sentiment_grad_2['model.'+key] = value
        model_4.load_state_dict(sentiment_grad_2, strict=False)
    
    model_4.eval()
    model_4.to('cuda')
    tokenizer_3 = model_4.tokenizer
    sentiment_classifier_2 = model_4.labels_classifier
    sentiment_classifier_2.load_state_dict(sent_classifier_grad_2)

    #########################################################
    """
    # If folder doesn't exist, then create it.
    MYDIR = ("/".join((args.output_log.split('/'))[:-1]))
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")

    if args.test == True:
        true_data = jsonlload("data/nikluge-sa-2022-test.jsonl")
        sentence_list_for_topic = []
        for tdata in true_data:
            sentence_list_for_topic.append([tdata['sentence_form']])
        
        dataset_1 = Custom_Dataset(sentence_list_for_topic, "eval", "trinary", tokenizer_0, args.input_length)
    else:
        dataset_1 = Custom_Dataset(args.eval_path, "valid", "trinary", tokenizer_0, args.input_length)

    print('Length of category validation data: ',len(dataset_1))
    loader_1 = DataLoader(dataset_1, batch_size=args.eval_batch_size, shuffle=False)

    first_out = []
    second_out = []
    third_out = []
    for batch in tqdm(iter(loader_1)):
        with torch.no_grad():
            output = model_0.model(
                batch['source_ids'].cuda(),
                batch['source_mask'].cuda()
            )
            output = dropout(output.pooler_output)
            output = first_classifier(output)
            output = torch.sigmoid(output)
            if sum(torch.where(output > 0.5, 1, 0)[0]) == 0:
                output[0][torch.argmax(output)] = 1
            first_out.append(output[0].detach().cpu())

    first_out = torch.stack(first_out)
    first_out = torch.where(first_out > 0.5, 1, 0)

    if args.test == False:
        first_dataset = pd.read_csv(args.eval_path, encoding='utf-8')
        for idx, row in first_dataset.iterrows():
            if first_out[idx][0] == 1:
                third_out.append([row['input'], '품질'])
            if first_out[idx][1] == 1:
                third_out.append([row['input'], '일반'])
            if first_out[idx][2] == 1:
                second_out.append([row['input']])
    else:
        for idx in range(len(sentence_list_for_topic)):
            if first_out[idx][0] == 1:
                third_out.append([sentence_list_for_topic[idx][0], '품질'])
            if first_out[idx][1] == 1:
                third_out.append([sentence_list_for_topic[idx][0], '일반'])
            if first_out[idx][2] == 1:
                second_out.append([sentence_list_for_topic[idx][0]])



    dataset_2_0 = Custom_Dataset(second_out, "eval", "category", tokenizer_2, args.input_length)
    print('Length of topic validation data: ',len(dataset_2_0))
    loader_2_0 = DataLoader(dataset_2_0, batch_size=args.eval_batch_size, shuffle=False)

    category_pred = []
    for batch in tqdm(iter(loader_2_0)):
        with torch.no_grad():
            output = model_2.model(
                batch['source_ids'].cuda(),
                batch['source_mask'].cuda()
            )
            output = dropout(output.pooler_output)
            output = category_classifier(output)
            output = torch.sigmoid(output)
            if sum(torch.where(output > 0.5, 1, 0)[0]) == 0:
                output[0][torch.argmax(output)] = 1
            category_pred.append(output[0].detach().cpu())
    
    category_pred = torch.stack(category_pred)
    category_pred = torch.where(category_pred > 0.5, 1, 0)

    for idx in range(len(second_out)):
        for ii in range(5):
            if category_pred[idx][ii] == 1:
                third_out.append([second_out[idx][0], category_list[ii]])

    dataset_2_1 = Custom_Dataset(third_out, "eval", "topic", tokenizer_1, args.input_length)
    print('Length of topic validation data: ',len(dataset_2_1))
    loader_2_1 = DataLoader(dataset_2_1, batch_size=args.eval_batch_size, shuffle=False)
    
    topic_pred = []
    for batch in tqdm(iter(loader_2_1)):
        with torch.no_grad():
            output = model_1.model(
                batch['source_ids'].cuda(),
                batch['source_mask'].cuda()
            )
            output = dropout(output.pooler_output)
            output = topic_classifier(output)
            output = torch.sigmoid(output)
            if sum(torch.where(output > 0.5, 1, 0)[0]) == 0:
                output[0][torch.argmax(output)] = 1
            topic_pred.append(output[0].detach().cpu())
    
    topic_pred = torch.stack(topic_pred)
    topic_pred = torch.where(topic_pred > 0.5, 1, 0)
    
    sentiment_list = []

    for idd in range(len(third_out)):
        for ii in range(4):
            if topic_pred[idd][ii] == 1:
                sentiment_list.append([third_out[idd][0], f"{topic_list[ii]}#{third_out[idd][1]}"])

    dataset_3 = Custom_Dataset(sentiment_list, "eval", "sentiment", tokenizer_3, args.input_length)
    print('Length of sentiment validation data: ',len(dataset_3))
    loader_3 = DataLoader(dataset_3, batch_size=args.eval_batch_size, shuffle=False)

    sentiment_pred = []

    for batch in tqdm(iter(loader_3)):
        with torch.no_grad():
            output = model_3.model(
                batch['source_ids'].cuda(),
                batch['source_mask'].cuda()
            )
            output = dropout(output.pooler_output)
            output = sentiment_classifier(output)
            output = torch.sigmoid(output)[0]
            value = int(torch.argmax(output).detach().cpu())
            if value == 0:
                sentiment_pred.append("positive")
            elif value == 1:
                sentiment_pred.append("negative")
            else:
                sentiment_pred.append("neutral")

    final_output_dict = {}
    for idd in range(len(sentiment_pred)):
        if sentiment_list[idd][0] not in final_output_dict:
            final_output_dict[sentiment_list[idd][0]] = [[sentiment_list[idd][1], sentiment_pred[idd]]]
        else:
            if [sentiment_list[idd][1], sentiment_pred[idd]] not in final_output_dict[sentiment_list[idd][0]]:
                final_output_dict[sentiment_list[idd][0]].append([sentiment_list[idd][1], sentiment_pred[idd]])

    if args.test == False:
        true_data = jsonlload("data/nikluge-sa-2022-dev.jsonl")
        pred_data = []
        cnt = 0
        for data in true_data:
            annotation_list = []
            if data['sentence_form'] in final_output_dict:
                annotation_list = final_output_dict[data['sentence_form']]
                cnt += 1
            sample_dict = {
                'id':data['id'],
                'sentence_form':data['sentence_form'],
                'annotation':annotation_list
            }
            pred_data.append(sample_dict)
        
        outfile_name = "output_file/dev_result_8.jsonl"
        with open(outfile_name , encoding= "utf-8" ,mode="w") as file: 
            for i in pred_data: 
                file.write(json.dumps(i,ensure_ascii=False) + "\n")
        print(evaluation_f1(true_data, pred_data))
    
    else:
        true_data = jsonlload("data/nikluge-sa-2022-test.jsonl")
        pred_data = []
        cnt = 0
        for data in true_data:
            annotation_list = []
            if data['sentence_form'] in final_output_dict:
                annotation_list = final_output_dict[data['sentence_form']]
                cnt += 1
            sample_dict = {
                'id':data['id'],
                'sentence_form':data['sentence_form'],
                'annotation':annotation_list
            }
            pred_data.append(sample_dict)
        outfile_name = "output_file/test_result_8.jsonl"
        with open(outfile_name , encoding= "utf-8" ,mode="w") as file: 
            for i in pred_data: 
                file.write(json.dumps(i,ensure_ascii=False) + "\n")
    