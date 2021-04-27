# -*- coding: utf-8 -*-
"""

"""

import utils

import torch
import torch.optim as optim
from model import Seq2Seq

from torch.utils.tensorboard import SummaryWriter

import tqdm
import random
import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

dataset = utils.load_pickled_data("./data/datasets/train_dataset.pkl")
val_dataset = utils.load_pickled_data("./data/datasets/val_dataset.pkl")

weights = utils.load_pickled_data("./data/glove/glove_embedd.pkl")

model = Seq2Seq(config.EMBEDD_DIM, config.HIDDEN_DIM, torch.tensor(weights).float())

model.to(device)


optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

model.train()

min_val_loss = 0
for epoch in tqdm.tqdm(range(config.EPOCHS)):
    
    for idx, batch in enumerate(dataset):
        
        input_tensor = torch.tensor(batch["text_ids"]).to(device)
        text_ids_ext = torch.tensor(batch["text_ids_ext"]).to(device)
        target_tensor = torch.tensor(batch["summary_ids"]).to(device)
        input_lengths = torch.tensor(batch["text_ids_len"])
        target_tensor_len = torch.tensor(batch["summary_ids_len"]).to(device)
        oovs = batch["oovs"]
        
        optimizer.zero_grad()

        out, loss = model(input_tensor,input_lengths, target_tensor,target_tensor_len, text_ids_ext, oovs)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        
        writer.add_scalar("Loss/train", loss.item(), epoch*config.BATCH_SIZE + idx)
        
        writer.flush()
        
        if idx % 1000 == 0: 
            model.eval()
            val_losses = 0
            for batch in val_dataset:
                input_tensor = torch.tensor(batch["text_ids"]).to(device)
                text_ids_ext = torch.tensor(batch["text_ids_ext"]).to(device)
                target_tensor = torch.tensor(batch["summary_ids"]).to(device)
                input_lengths = torch.tensor(batch["text_ids_len"])
                target_tensor_len = torch.tensor(batch["summary_ids_len"]).to(device)
                oovs = batch["oovs"]
   
 
                out, loss = model(input_tensor,input_lengths, target_tensor,target_tensor_len, text_ids_ext, oovs)
                val_losses += loss.item()
    
            loss = val_losses / len(val_dataset)
            if loss < min_val_loss:
                torch.save(model.state_dict(), f"./weights/weights_{loss:.3f}_{epoch*config.BATCH_SIZE + idx}.pt")
                min_val_loss = loss
            writer.add_scalar("Loss/Validation", loss, epoch*config.BATCH_SIZE + idx)
            writer.flush()
            model.train()
    
    random.shuffle(dataset)


torch.save(model.state_dict(), "model_weights.pt")

