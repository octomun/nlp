import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb
import os

def CELoss(pred_outs, labels):
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs,labels)
    return loss_val

def SaveModel(model_info, path, model_name):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model_info, os.path.join(path, model_name))

from dataset import post_loader
from model import PostModel

data_path = '../DATA/korean_smile_style_dataset/smilestyle_dataset.tsv'
model_save_path = '../MODEL/'
post_model = PostModel().cuda()
post_dataset = post_loader(data_path)
post_dataloader = DataLoader(post_dataset, batch_size=2, shuffle= False, collate_fn=post_dataset.collate_fn)

training_epochs = 1
max_grad_norm = 10
lr = 1e-5
num_traning_steps = len(post_dataset)*training_epochs
num_warmup_steps = len(post_dataset)
optimizer = torch.optim.AdamW(post_model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_traning_steps, num_training_steps=num_traning_steps)

for epoch in range(training_epochs):
    post_model.train()
    for i_batch, data in enumerate(tqdm (post_dataloader)):
        batch_corrupt_tokens, batch_output_tokens, batch_corrupt_mask_positions, batch_urc_inputs, batch_urc_labels, batch_mlm_attentions, batch_urc_attentions = data
        batch_corrupt_tokens = batch_corrupt_tokens.cuda()
        batch_output_tokens = batch_output_tokens.cuda()
        batch_urc_inputs = batch_urc_inputs.cuda()
        batch_urc_labels = batch_urc_labels.cuda()
        batch_mlm_attentions = batch_mlm_attentions.cuda()
        batch_urc_attentions = batch_urc_attentions.cuda()

        """Prediction"""
        corrupt_mask_outputs, urc_cls_outputs = post_model(batch_corrupt_tokens, batch_corrupt_mask_positions, batch_urc_inputs, batch_mlm_attentions, batch_urc_attentions)
        """Loss calculation & training"""
        original_token_indexs = []
        for i_batch in range (len (batch_corrupt_mask_positions)):
            original_token_index = []
            batch_corrupt_mask_position = batch_corrupt_mask_positions[i_batch]
            for pos in batch_corrupt_mask_position:
                original_token_index.append(batch_output_tokens[i_batch, pos].item())
            original_token_indexs.append(original_token_index)
        mlm_loss = 0
        for corrupt_mask_output, original_token_index in zip(corrupt_mask_outputs, original_token_indexs):
            mlm_loss += CELoss (corrupt_mask_output, torch.tensor (original_token_index). cuda ( ) )
        urc_loss = CELoss (urc_cls_outputs, batch_urc_labels)
        loss_val= mlm_loss + urc_loss
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(post_model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad ()
        model_info = {'epoch' : epoch, 'model' : post_model.state_dict(), 'optimizer' : optimizer, 'loss' : loss_val}
SaveModel(model_info, path=model_save_path, model_name=f'PostModel_{epoch}.bin')
