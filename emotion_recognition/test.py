from cmath import log
from email.policy import default
import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

import os
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
import pdb

import logging

from model import ERC_model

from dataset import data_loader
from torch.utils.data import DataLoader

def model_test(args):

    # 로그 생성
    logger = logging.getLogger()

    # 로그의 출력 기준 설정
    logger.setLevel(logging.INFO)

    # log 출력
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    # log를 파일에 출력
    file_handler = logging.FileHandler('erc_test.log', encoding='utf-8')
    logger.addHandler(file_handler)

    test_dataset = data_loader('./MELD/data/MELD/test_sent_emo.csv')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)


    clsNum = len(test_dataset.emoList)
    erc_model = ERC_model(clsNum).cuda()

    logger.info("############테스트 시작############")
    best_dev_fscore = 0
    save_path = args.save_path
    model_name = args.model_name
    check_point = torch.load(os.path.join(save_path, model_name))
    erc_model.load_state_dict(check_point['model_state_dict'])
    erc_model.eval()
    correct = 0
    label_list = []
    pred_list = []    
    
    error_samples = []
    class_acc = dict.fromkeys(test_dataset.emoList,0)
    class_label_count = dict.fromkeys(test_dataset.emoList,0)
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(test_dataloader)):
            """Prediction"""
            batch_padding_token, batch_padding_attention_mask, batch_PM_input, batch_label = data
            batch_padding_token = batch_padding_token.cuda()
            batch_padding_attention_mask = batch_padding_attention_mask.cuda()
            batch_PM_input = [[x2.cuda() for x2 in x1] for x1 in batch_PM_input]
            batch_label = batch_label.cuda()        

            """Prediction"""
            pred_logits = erc_model(batch_padding_token, batch_padding_attention_mask, batch_PM_input)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).item()
            true_label = batch_label.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)            
            if pred_label != true_label:
                error_samples.append([batch_padding_token, true_label, pred_label])
                logger.info('---------------------------------------------------------------------------')
                logger.info('data : ' +test_dataset.tokenizer.decode(batch_padding_token.squeeze(0).tolist()))
                logger.info('label : ' + test_dataset.emoList[true_label])
                logger.info('pred : ' + test_dataset.emoList[pred_label])
            if pred_label == true_label:
                correct += 1
                class_acc[test_dataset.emoList[true_label]] += 1
            class_label_count[test_dataset.emoList[true_label]] += 1
        acc = correct/len(test_dataloader)
        

    return error_samples, acc, [class_acc, class_label_count], pred_list, label_list