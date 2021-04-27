# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn as nn

import torch.nn.functional as F


class Loss(nn.Module):
    
    def __init__(self, pad_index, coverage_weight):
        
        super().__init__()
        self.pad_index = pad_index
        self.coverage_weight = coverage_weight
        
    
    def forward(self, output, target, target_tensor_len):
        
        final_dist = output["final_dist"]
        
        log_probs = torch.log(final_dist)
        nll_loss = F.nll_loss(log_probs, target, ignore_index=self.pad_index, reduction="mean")
        
        
        attn_dist = output["attn_dist"]
        coverage = output["coverage"]
        
        min_val = torch.min(attn_dist, coverage)
        coverage_loss = torch.sum(min_val, dim=1)
        
        
        decoder_mask = target != 0
        
        coverage_loss = coverage_loss.masked_fill(~decoder_mask, 0.0)
        
        coverage_loss = torch.sum(coverage_loss) / torch.sum(target_tensor_len)
        
        loss =  nll_loss + self.coverage_weight * coverage_loss
        return loss