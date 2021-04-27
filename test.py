# -*- coding: utf-8 -*-
"""

"""

import utils

import torch
import torch.optim as optim
from model import Seq2Seq

from torch.utils.tensorboard import SummaryWriter
from Vocab import Vocabulary
import tqdm

import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

dataset = utils.load_pickled_data("./data/datasets/test_dataset.pkl")


vocab = Vocabulary.load_vocabulary("./data/vocab/vocab.pkl")


weights = utils.load_pickled_data("./data/glove/glove_embedd.pkl")

model = Seq2Seq(config.EMBEDD_DIM, config.HIDDEN_DIM, torch.tensor(weights).float())

model.to(device)

model.load_state_dict(torch.load("./model_weights/model_weight.pt", map_location=device))


model.eval()

predictions = []
targets = []
test_loss = 0
for idx, batch in enumerate(tqdm.tqdm(dataset)):
    
    input_tensor = torch.tensor(batch["text_ids"]).to(device)
    text_ids_ext = torch.tensor(batch["text_ids_ext"]).to(device)
    target_tensor = torch.tensor(batch["summary_ids"]).to(device)
    input_lengths = torch.tensor(batch["text_ids_len"])
    target_tensor_len = torch.tensor(batch["summary_ids_len"]).to(device)
    oovs = batch["oovs"]
    

    out, loss = model(input_tensor,input_lengths, target_tensor,target_tensor_len, text_ids_ext, oovs)
    
    test_loss += loss.item()
    for i in range(out["final_dist"].shape[0]):
        final = out["final_dist"][i, :, :]
        
        output = []
        
        for index in range(final.shape[1]):
            
            
            output.append(torch.argmax(final[:, index]).item())
            
        
        words = vocab.output_to_words(output, oovs[i])
        words_tgt = vocab.output_to_words(batch["summary_ids"][i], oovs[i])
        
        tgt = " ".join(words_tgt)
        inf = " ".join(words)

        predictions.append(inf)
        targets.append(tgt)
            
print(f"Average Test Loss over 10,000 samples: {test_loss / len(dataset)}")
with open("test_predictions.txt", "w") as outfile:
    for prediction in predictions:
        outfile.write(prediction + "\n")
    
with open("test_targets.txt", "w") as outfile:
    for target in targets:
        outfile.write(target + "\n")
    

