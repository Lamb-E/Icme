import numpy as np
import torch
import random
import time
from sklearn.metrics import roc_auc_score
import sys
from utils import *
import argparse
import pandas as pd
import pdb
import torch.nn as nn


class latent_guard(nn.Module):
    def __init__(self, ):
        super(latent_guard, self).__init__()
        num_heads = 16;
        head_dim = 32;
        out_dim = 128
        self.model = EmbeddingMappingLayer(num_heads, head_dim, out_dim).to('cuda')
        self.model.load_state_dict(torch.load("/home/zcy/attack/WWW_rebuttal/LatentGuard/model_parameters.pth"))
        self.model.eval()

        target_concept_set = train_concepts
        target_concept_set = list(target_concept_set)

        print('Preparing concept embeddings... it may take seconds...')
        concept_embs = [wrapClip.get_emb(concept).to(device) for concept in target_concept_set]
        print('Concept embeddings prepared.')

        all_concept_emb = torch.cat(concept_embs, dim=0).to(device)
        self.all_concept_emb = all_concept_emb[:, 0, :]

    def forward(self, prompt):
        prompt_emb = wrapClip.get_emb(prompt).to(device)

        with torch.no_grad():
            prompt_emb = prompt_emb.to(device)
            repeated_prompt_emb = prompt_emb.repeat(len(train_concepts), 1, 1)
            output = self.model(repeated_prompt_emb.to(device), self.all_concept_emb.to(device))
            dot_product = forward_contra_model(self.model, output)

            predicted_maxv = dot_product.max(0)[0].cpu().numpy()
            pred = np.array(predicted_maxv)
            pred_labels = (pred >= 9.0131).astype(int)
        return pred_labels