# -*- coding: utf-8 -*-
"""

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import utils

from loss import Loss

import config

class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size=512):
        
        super().__init__()
        
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, hidden_size)
        self.relu = nn.LeakyReLU()
        
    
    def forward(self, input_tensor, input_lengths):
        
        
        # seq_length x b x embedd_dim
        
        x = nn.utils.rnn.pack_padded_sequence(input_tensor,input_lengths, batch_first=True, enforce_sorted=False)
        
        output, hidden = self.rnn(x)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=-1)
        
        hidden = self.relu(self.linear(hidden))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return output, hidden
        
        
        
        
class Attention(nn.Module):
    
    def __init__(self, hidden_size):
        
        super().__init__()
        
        self.W_h = nn.Linear(hidden_size * 2, hidden_size)
        self.W_s = nn.Linear(hidden_size, hidden_size)
        
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax()
        
    def forward(self, decoder_output, encoder_output, mask):
        
        
        decoder_output = decoder_output.unsqueeze(1)
        x = torch.tanh(self.W_h(encoder_output) + self.W_s(decoder_output))
        
        x = self.v(x)
        x = x.squeeze()
        
        masked_scores = x.masked_fill(~mask, float('-inf'))
        attn_dist = F.softmax(masked_scores, dim=-1)
        
        return attn_dist


class Decoder(nn.Module):
    
    def __init__(self, embed_dim, hidden_size, vocab_size):
        
        super().__init__()
        self.rnn = nn.GRUCell(embed_dim, hidden_size)
        self.attention = Attention(hidden_size)
        
        self.v = nn.Linear(hidden_size * 3, hidden_size)
        self.v_hat = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, decoder_input, previous_hidden_state, encoder_output, mask):
        
        hidden = self.rnn(decoder_input, previous_hidden_state)

        attn_dist = self.attention(hidden, encoder_output, mask)
        
        context_vector = torch.bmm(attn_dist.unsqueeze(1), encoder_output)
        context_vector = torch.sum(context_vector, dim=1)
        
        
        out = self.v_hat(self.v(torch.cat((hidden, context_vector), dim=-1)))
        
        vocab_dist = F.softmax(out, dim=-1)
        return vocab_dist, attn_dist, context_vector, hidden
        
        
        
        

class Seq2Seq(nn.Module):
    
    def __init__(self, embed_dim, hidden_size, vocab, pretrained_embeddings=None):
        
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings)
        # self.embedding = nn.Embedding(20000, 100, 0)
        
        embed_dim = self.embedding.embedding_dim
        
        self.encoder = Encoder(embed_dim, hidden_size)
        self.decoder = Decoder(embed_dim, hidden_size, config.VOCAB_SIZE)
        
        self.attention = Attention(hidden_size)
        
        self.w_h = nn.Linear(hidden_size * 2, 1)
        self.w_s = nn.Linear(hidden_size, 1)
        self.w_x = nn.Linear(embed_dim, 1)
        
        self.criterion = Loss(vocab.pad_index, config.COVERAGE_WEIGHT)
        
        self.vocab = vocab
        

    def forward(self, article_tensor, article_lengths, summary_tensor, target_tensor_len, text_ids_ext, oovs, teacher_force_ratio=0):
        
        final_dists = []
        attn_dists = []
        coverages = []
        
        batch_size = article_tensor.shape[0]
        
        encoder_input_embedding = self.embedding(article_tensor)
        
        encoder_output, hidden = self.encoder(encoder_input_embedding, article_lengths)
        
        # <SOS> Tokens
        decoder_input = article_tensor[:, 0]
        
        mask = article_tensor != 0
        
        coverage = torch.zeros_like(article_tensor, device=article_tensor.device).float()
        
        
        for sequence_index in range(config.MAX_TARGET_LENGTH):
            
            decoder_input_embedding = self.embedding(decoder_input)
            
            
            decoder_output = self.decoder(decoder_input_embedding, hidden, encoder_output, mask)

            vocab_dist, attn_dist, context_vector, hidden  = decoder_output
            
            coverage = coverage + attn_dist
            
            
            context_features = self.w_h(context_vector)
            decoder_features = self.w_s(hidden)
            input_features = self.w_x(decoder_input_embedding)
            
            p_gen = torch.sigmoid(context_features + decoder_features + input_features) 
            
            
            vocab_dist = p_gen * vocab_dist
            weighted_attn_dist = (1.0 - p_gen) * attn_dist
            
            max_oov_length = max(map(len, oovs))
            extra_zeros = torch.zeros((batch_size, max_oov_length), device=vocab_dist.device)
            
            extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)

            final_dist = extended_vocab_dist.scatter_add(
                dim=-1,
                index=text_ids_ext,
                src=weighted_attn_dist
            )
            
            final_dists.append(final_dist)
            attn_dists.append(attn_dist)
            coverages.append(coverage)
            
            teacher_force = random.random() < teacher_force_ratio
            
            top1 = final_dist.argmax(1) 
            
            decoder_input = summary_tensor[:, sequence_index] if teacher_force else top1
            
            decoder_input = torch.where(decoder_input >= self.vocab.vocab_size, self.vocab.pad_index, decoder_input)

            
        
        final_dists = torch.stack(final_dists, dim=-1)
        attn_dists = torch.stack(attn_dists, dim=-1)    
        coverages = torch.stack(coverages, dim=-1)

        out =  {
            'final_dist': final_dists,
            'attn_dist': attn_dists,
            'coverage': coverages
        }


        loss = self.criterion(out, summary_tensor, target_tensor_len)
        
        return out, loss

if __name__ == "__main__":
    
    batch_size = 4
    seq_length = 10
    input_tensor = torch.randint(0, 30, (4, 20))
    target_tensor = torch.randint(0, 30, (4, 5))
    
    input_lengths = [20, 16, 15, 13]
    
    model = Seq2Seq(100, 512, None)
    
    batches = utils.load_pickled_data("./testdataset.pkl")
    for batch in batches:
        
        
        input_tensor = torch.tensor(batch["text_ids"])
        text_ids_ext = torch.tensor(batch["text_ids_ext"])
        target_tensor = torch.tensor(batch["summary_ids"])
        input_lengths = torch.tensor(batch["text_ids_len"])
        target_tensor_len = torch.tensor(batch["summary_ids_len"])
        oovs = batch["oovs"]

        out = model(input_tensor,input_lengths, target_tensor, target_tensor_len, text_ids_ext, oovs)
        
    
    
    
# weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3], [5, 7.1, 6.3], [1, 5.1, 9.3]])
# embedding = nn.Embedding.from_pretrained(weight)

# input = torch.LongTensor([[1, 2], [0, 3], [1,3], [2,3]])
# out = embedding(input)
# print(out)

# print()
# inputT = torch.transpose(input, 0, 1)
# outT = embedding(input)
# print(outT)