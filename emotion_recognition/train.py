from email.policy import default
import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

import os
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
import pdb

import logging

def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def SaveModel(model_info : dict, path, model_name):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({'epoch': model_info['epoch'],
                'model_state_dict' : model_info['model'].state_dict(),
                'optimizer_state_dict': model_info['optimizer'].state_dict(),
                'loss': model_info['loss'],
                }, os.path.join(path, model_name))
    
def CalACC(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            """Prediction"""
            batch_padding_token, batch_padding_attention_mask, batch_PM_input, batch_label = data
            batch_padding_token = batch_padding_token.cuda()
            batch_padding_attention_mask = batch_padding_attention_mask.cuda()
            batch_PM_input = [[x2.cuda() for x2 in x1] for x1 in batch_PM_input]
            batch_label = batch_label.cuda()        

            """Prediction"""
            pred_logits = model(batch_padding_token, batch_padding_attention_mask, batch_PM_input)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).item()
            true_label = batch_label.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
        acc = correct/len(dataloader)
    return acc, pred_list, label_list

from dataset import data_loader
from torch.utils.data import DataLoader

def model_train(args):

    # ?????? ??????
    logger = logging.getLogger()

    # ????????? ?????? ?????? ??????
    logger.setLevel(logging.INFO)

    # log ??????
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    # log??? ????????? ??????
    file_handler = logging.FileHandler('erc.log', encoding='utf-8')
    logger.addHandler(file_handler)

    train_dataset = data_loader('./MELD/data/MELD/train_sent_emo.csv')
    dev_dataset = data_loader('./MELD/data/MELD/dev_sent_emo.csv')
    test_dataset = data_loader('./MELD/data/MELD/test_sent_emo.csv')

    batch_size = args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)  # num_workers=4,
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn)

    from model import ERC_model
    clsNum = len(train_dataset.emoList)
    erc_model = ERC_model(clsNum).cuda()

    """ ????????? ??????????????? """
    training_epochs = args.epochs
    max_grad_norm = 10
    lr = args.lr
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(erc_model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    logger.info("############?????? ??????############")
    best_dev_fscore = 0
    save_path = args.save_path
    model_name = args.model_name
    
    for epoch in tqdm(range(training_epochs)):
        erc_model.train() 
        logger.info(f"epoch : {epoch}")
        for i_batch, data in enumerate(tqdm(train_dataloader)):
            batch_padding_token, batch_padding_attention_mask, batch_PM_input, batch_label = data
            batch_padding_token = batch_padding_token.cuda()
            batch_padding_attention_mask = batch_padding_attention_mask.cuda()
            batch_PM_input = [[x2.cuda() for x2 in x1] for x1 in batch_PM_input]
            batch_label = batch_label.cuda()        
            
            """Prediction"""
            pred_logits = erc_model(batch_padding_token, batch_padding_attention_mask, batch_PM_input)
            
            """Loss calculation & training"""
            loss_val = CELoss(pred_logits, batch_label)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(erc_model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        """Dev & Test evaluation"""
        erc_model.eval()
        
        dev_acc, dev_pred_list, dev_label_list = CalACC(erc_model, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        
        logger.info("Dev W-avg F1: {}".format(dev_fbeta))
        
        test_acc, test_pred_list, test_label_list = CalACC(erc_model, test_dataloader)
        """Best Score & Model Save"""
        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta

            test_acc, test_pred_list, test_label_list = CalACC(erc_model, test_dataloader)
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')                
            model_info = {'epoch' : epoch, 'model' : erc_model, 'optimizer' : optimizer, 'loss' : loss_val}
            SaveModel(model_info, save_path, model_name)
            logger.info("Epoch:{}, Test W-avg F1: {}".format(epoch, test_fbeta))