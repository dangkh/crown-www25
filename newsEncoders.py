import pickle
import math
# import numpy as np
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# import sys
from config import Config
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from layers import Conv1D, Conv2D_Pool, MultiHeadAttention, Attention, ScaledDotProduct_CandidateAttention, CandidateAttention

class NewsEncoder(nn.Module):
    def __init__(self, config: Config):
        super(NewsEncoder, self).__init__()
        self.word_embedding_dim = config.word_embedding_dim
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.category_embedding_dim)
        self.category_embedding.weight.requires_grad = False
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.subCategory_embedding_dim)
        self.subCategory_embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subCategory_embedding.weight[0])

    # Input
    # title_text          : [batch_size, news_num, max_title_length]   # [64, 5, 32]
    # title_mask          : [batch_size, news_num, max_title_length]
    # title_entity        : [batch_size, news_num, max_title_length]
    # content_text        : [batch_size, news_num, max_content_length]
    # content_mask        : [batch_size, news_num, max_content_length]
    # content_entity      : [batch_size, news_num, max_content_length]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # user_embedding      : [batch_size, user_embedding_dim]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        raise Exception('Function forward must be implemented at sub-class')

    # Input
    # news_representation : [batch_size, news_num, unfused_news_embedding_dim]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def feature_fusion(self, news_representation, category, subCategory):
        category_representation = self.category_embedding(category)                                                                                    # [batch_size, news_num, category_embedding_dim]
        subCategory_representation = self.subCategory_embedding(subCategory)                                                                           # [batch_size, news_num, subCategory_embedding_dim]
        news_representation = torch.cat([news_representation, self.dropout(category_representation), self.dropout(subCategory_representation)], dim=2) # [batch_size, news_num, news_embedding_dim]
        return news_representation


# Our proposed model: CIDER - news encoder
class CIDER(NewsEncoder):
    def __init__(self, config: Config):
        super(CIDER, self).__init__(config)

        self.max_title_length = config.max_title_length
        self.max_body_length = config.max_abstract_length
        
        # self.news_embedding_dim = config.head_num * config.head_dim         # for MH-Attention (400)
        self.news_embedding_dim = config.word_embedding_dim                   # for Transformer (300)
        
        # Transformer encoder
        # self.title_pos_encoder = PositionalEncoding(config.word_embedding_dim, config.dropout_rate, config.max_title_length)
        # self.body_pos_encoder = PositionalEncoding(config.word_embedding_dim, config.dropout_rate, config.max_abstract_length)
        # encoder_layers = TransformerEncoderLayer(config.word_embedding_dim, config.head_num, config.feedforward_dim, config.dropout_rate, batch_first=True)   # head_num은 word_embedding_dim을 나눌 수 있어야 함
        # self.transformer = TransformerEncoder(encoder_layers, config.num_layers)
        
        # MAB(Multihead Attention Block) encoder
        self.MAB = MAB(config.word_embedding_dim, 
                       config.word_embedding_dim, 
                       config.word_embedding_dim, 
                       num_heads = config.isab_num_heads, 
                       ln = True)
        
        # ISAB(Induced Set Attention Block) emcoder - 230915
        self.ISAB = ISAB(dim_in = config.word_embedding_dim, 
                         dim_out = config.word_embedding_dim,
                         num_heads = config.isab_num_heads,        # The number of ISAB heads       4,  choices=[2, 4, 8]
                         num_inds = config.isab_num_inds,          # The number of inducing points  4,  choices=[2, 4, 6, 8]
                         ln = True)
        
                # # Knowledge encoder
                # knowledge_indices = KnowledgePreparing()
                # self.knowledge_encoder = KnowledgeEncoding(config.vocabulary_size, config.word_embedding_dim, knowledge_indices, alpha, config.dropout_rate)
        
        # self.title_multiheadAttention = MultiHeadAttention(config.head_num, config.word_embedding_dim, config.max_title_length, config.max_title_length, config.head_dim, config.head_dim)          # for MH-Attention
        # self.body_multiheadAttention = MultiHeadAttention(config.head_num, config.word_embedding_dim, config.max_abstract_length, config.max_abstract_length, config.head_dim, config.head_dim)     # for MH-Attention
        # self.title_multiheadAttention = torch.nn.MultiheadAttention(config.word_embedding_dim, config.head_num, dropout=0.2, batch_first=True)
        # self.body_multiheadAttention = torch.nn.MultiheadAttention(config.word_embedding_dim, config.head_num, dropout=0.2, batch_first=True)
        
        self.title_attention = Attention(config.word_embedding_dim, config.attention_dim) 
        self.body_attention = Attention(config.word_embedding_dim, config.attention_dim)
        
        self.category_affine = nn.Linear(config.category_embedding_dim + config.subCategory_embedding_dim, config.category_embedding_dim)               
        
        self.affine1 = nn.Linear(config.intent_embedding_dim, config.attention_dim, bias=True)               # for intent attention (linear transformation)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)                                        # for intent attention (distribution score)
        
        self.intent_num = config.intent_num     # hyperparameter k
        self.intent_layers = nn.ModuleList([nn.Linear(config.word_embedding_dim
                                                      + config.category_embedding_dim
                                                      , config.intent_embedding_dim, bias=True) 
                                            for _ in range(self.intent_num)])
    
    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.body_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)
        # Initialize each intent layer with different weights to learn different embedding for each intent
        for intent_layer in self.intent_layers:
            nn.init.xavier_uniform_(intent_layer.weight)
            nn.init.zeros_(intent_layer.bias)

    # Input
    # title/body_embedding : [batch_size, news_num, title/body_embedding_dim]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # Output
    # category-aware_title/body_embedding : [batch_size, news_num, title/body_embedding_dim + (category_dim + subCategory_dim) // 2]
    
    def category_aware_represent(self, news_embedding, category, subCategory):        
        # 하나로 합치는데, category + sub_cate concat해서 linear 100->50
        category_representation = self.category_affine(torch.cat([self.category_embedding(category), 
                                                                  self.subCategory_embedding(subCategory)], dim=2))  # [batch_size, news_num, category+subCategory_embedding_dim]                
        category_aware_embedding = torch.cat([news_embedding, self.dropout(category_representation)], dim=2)         # [batch_size, news_num, news_embedding_dim = title/body + category] 300+50 = 350
        # category_aware_embedding = torch.cat([news_embedding, category_representation], dim=2)
        return category_aware_embedding
    
    # Input: [batch_size, news_num, news_embedding_dim]
    # Output: [batch_size, news_num, intent_embedding_dim] * k
    # Apply k-FC layer for k-intent disentanglement
    def k_intent_disentangle(self, intent_num, news_embedding):                                 
        k_intent_embeddings = []
        for i in range(intent_num):
            # Apply different linear transformations for each intent
            intent_embedding = F.relu(self.intent_layers[i](news_embedding), inplace=True)      
            k_intent_embeddings.append(intent_embedding)
        
        return k_intent_embeddings                                                              
    
    def intent_attention(self, intent_num, k_intent_embeddings):
        k_intent_embeddings = k_intent_embeddings[:intent_num]
        feature = torch.stack(k_intent_embeddings, dim=2)                                                   # [batch_size, news_num, k, intent_embedding_dim]
        # Apply attention mechanism to calculate attention score(intent distribution) -> weighted attention
        intent_distribution = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)             # [batch_size, news_num, 1]
        intent_embedding = (feature * intent_distribution).sum(dim=2, keepdim=False)                        # [batch_size, news_num, intent_embedding_dim]
        
        return intent_embedding, intent_distribution                                                        # [batch_size, news_num, intent_embedding_dim], [batch_size, news_num, 1]

    def similarity_compute(self, title_intent_distribution, body_intent_distribution):                            # [batch_size, news_num, 1]
        cosine_similarity = F.cosine_similarity(title_intent_distribution, body_intent_distribution, dim=2)
        # similarity normalization
        threshold = 0.5
        title_body_similarity = (cosine_similarity + 1) / 2.0
        min_similarity = torch.min(title_body_similarity)
        max_similarity = torch.max(title_body_similarity)
        scaled_title_body_similarity = torch.where(title_body_similarity == 0, 
                                                   threshold,
                                                   (title_body_similarity - min_similarity) / (max_similarity - min_similarity))
        
        return scaled_title_body_similarity                                                                        # [batch_size, news_num, 1]
    
    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)         # 배치 사이즈 (유저 수)
        news_num = title_text.size(1)           # 유저 당 학습하는 뉴스의 수 (positive_sample_num(본 뉴스) + negative_sample_num(안 본 뉴스) = 1 + 4 = 5)
                                                # 
        batch_news_num = batch_size * news_num  # 배치 당 총 뉴스의 수 (유저 * 유저 당 뉴스의 수)
        
        t_mask = title_mask.view([batch_news_num, self.max_title_length])                                   # [batch_size * news_num, max_title_length]
        b_mask = content_mask.view([batch_news_num, self.max_body_length])                                  # [batch_size * news_num, max_body_length]
        
        # (1) Word embedding
        # to convert a news title from a word sequence into a sequence of dense semantic vectors
        # 여기에서 max_title_length 잘랐는지 아닌지 확실히 check
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])          # [batch_size * news_num, max_title_length, word_embedding_dim]
        body_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_body_length, self.word_embedding_dim])          # [batch_size * news_num, max_content_length, word_embedding_dim]
        
        # (2) Multi-head Attention(MA) encoding
        # to learn contextual representations of words by capturing their interactions
        # (such long-distance interactions usually can not be captured by CNN)
        # title_m, _ = self.title_multiheadAttention(title_w, title_w, title_w)                  # [batch_size * news_num, max_title_length, news_embedding_dim]
        # body_m, _ = self.body_multiheadAttention(body_w, body_w, body_w)                       # [batch_size * news_num, max_content_length, news_embedding_dim]
        
                # ## Knowledge encoding
                # residual = title_w
                # title_k = self.knowledge_encoder(title_w, title_text)                                   # [batch_size * news_num, max_title_length, word_embedding_dim]
                # title_k += residual                                                                     # [batch_size * news_num, max_title_length, word_embedding_dim]
        
        # (2) Transformer encoding
        # title_p = self.title_pos_encoder(title_w)                                                       # [batch_size * news_num, max_title_length, news_embedding_dim]
        # title_t = self.transformer(title_p)                                                             # [batch_size * news_num, max_title_length, news_embedding_dim]
        # title_embedding = title_t.mean(dim=1).view([batch_size, news_num, self.news_embedding_dim])     # [batch_size, news_num, news_embedding_dim] 300              for Transformer(average)
        
        # body_p = self.body_pos_encoder(body_w)                                                          # [batch_size * news_num, max_content_length, news_embedding_dim]
        # body_t = self.transformer(body_p)                                                               # [batch_size * news_num, max_content_length, news_embedding_dim]
        # body_embedding = body_t.mean(dim=1).view([batch_size, news_num, self.news_embedding_dim])       # [batch_size, news_num, news_embedding_dim] 300              for Transformer(average)
        
        # (2) MAB(Multihead Attention Block) encoding
        title_mab1 = self.dropout(self.MAB(title_w, title_w))                                                # [batch_size * news_num, max_title_length, news_embedding_dim]
        title_mab2 = self.dropout(self.MAB(title_mab1, title_mab1))                                          # [batch_size * news_num, max_title_length, news_embedding_dim]
        
        title_embedding = title_mab2.mean(dim=1).view([batch_size, news_num, self.news_embedding_dim])       # [batch_size, news_num, news_embedding_dim]
        
        body_mab1 = self.dropout(self.MAB(body_w, body_w))                                                   # [batch_size * news_num, max_title_length, news_embedding_dim]
        body_mab2 = self.dropout(self.MAB(body_mab1, body_mab1))                                             # [batch_size * news_num, max_title_length, news_embedding_dim]
        
        body_embedding = body_mab2.mean(dim=1).view([batch_size, news_num, self.news_embedding_dim])         # [batch_size, news_num, news_embedding_dim]
                
        
        # (2) ISAB(Induced Set Attention Block) encoding - 230915
        # title_isab1 = self.dropout(self.ISAB(title_w))                                                       # [batch_size * news_num, max_title_length, news_embedding_dim]
        # title_isab2 = self.dropout(self.ISAB(title_isab1))                                                   # [batch_size * news_num, max_title_length, news_embedding_dim]

        # title_embedding = title_isab2.mean(dim=1).view([batch_size, news_num, self.news_embedding_dim])      # [batch_size, news_num, news_embedding_dim]

        # body_isab1 = self.dropout(self.ISAB(body_w))                                                         # [batch_size * news_num, max_title_length, news_embedding_dim]
        # body_isab2 = self.dropout(self.ISAB(body_isab1))                                                     # [batch_size * news_num, max_title_length, news_embedding_dim]

        # body_embedding = body_isab2.mean(dim=1).view([batch_size, news_num, self.news_embedding_dim])        # [batch_size, news_num, news_embedding_dim]
        
        
        # (3) Word-level Attention encoding (* instead of title_t.mean(dim=1))
        # to select important words in news titles to learn more informative news representations        
        
        # title_embedding = self.title_attention(title_m, mask=t_mask).view([batch_size, news_num, self.news_embedding_dim])                  # [batch_size, news_num, news_embedding_dim]      for MH-Attention
        # body_embedding = self.body_attention(body_m, mask=b_mask).view([batch_size, news_num, self.news_embedding_dim])                     # [batch_size, news_num, news_embedding_dim]      for MH-Attention
        
        # title_embedding = self.title_attention(title_t, mask=t_mask).view([batch_size, news_num, self.news_embedding_dim])                  # [batch_size, news_num, news_embedding_dim]      for Transformer(attention)
        # body_embedding = self.body_attention(body_t, mask=b_mask).view([batch_size, news_num, self.news_embedding_dim])                     # [batch_size, news_num, news_embedding_dim]      for Transformer(attention)
        
        
        # (4) Category-aware intent disentanglement
        ###
        ### 1. input: Category embedding, Title(Body) embedding -> concat 
        ### -> output: Category-Title(Body) embedding
        category_title_embedding = self.category_aware_represent(title_embedding, category, subCategory)     # [batch_size, news_num, news(title)_embedding_dim] 300+50=350
        category_body_embedding = self.category_aware_represent(body_embedding, category, subCategory)        # [batch_size, news_num, news(body)_embedding_dim]
        
        ### 2. input: each C-T/C-B embedding -> k-linear layers(= k-intent layers for disentanglement) 
        ### -> output: k-category-aware C-T/C-B intent embeddings (# of intents = # of fully connencted layers)
        # intent dim 64, 128, 256, 512 변경 가능 (지금은 400, 이게 곧 news_representation dim)
        k = self.intent_num
        title_k_intent_embeddings = self.k_intent_disentangle(k, category_title_embedding)              # [batch_size, news_num, intent_embedding_dim] * k 
        body_k_intent_embeddings = self.k_intent_disentangle(k, category_body_embedding)                # [batch_size, news_num, intent_embedding_dim] * k 
        
        # (5) Intent-distribution based Attention
        ### 3. input: k-category-aware C-T/C-B intent embeddings -> Attention layer(MHSA or MAB), att_score = intent distribution 
        ### -> output: Title/body embedding, Title/body intent distribution for similarity computation
        title_intent_embedding, title_intent_distribution = self.intent_attention(k, title_k_intent_embeddings)                    # [batch_size, news_num, intent_embedding_dim]
        body_intent_embedding, body_intent_distribution = self.intent_attention(k, body_k_intent_embeddings)                      # [batch_size, news_num, intent_embedding_dim]
        
        # (6) Title-Body similarity computation (todo)
        # 두 distribution의 cosine or jaccard or euclidean distance similarity 계산 
        ### 4. input: Title/body intent distribution -> similarity computation
        ### -> output: similarity score (0~1)
        title_body_similarity = self.similarity_compute(title_intent_distribution, body_intent_distribution) # [batch_size, news_num, 1]
        # print('title_body_similarity: ', title_body_similarity)
        # exit()
        
        ### 5. input: Title embedding, Body embedding -> average (weighted sum)
        ### -> output: news representation embedding
        # news_representation = torch.cat([title_intent_embedding, title_body_similarity * body_intent_embedding], dim=2)      # [batch_size, news_num, title+body intent_embedding_dim]     # concat 
        news_representation = title_intent_embedding + title_body_similarity * body_intent_embedding                         # [batch_size, news_num, title(body) intent_embedding_dim]     # average
        
        return news_representation

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


# Collaborative News Encoding(CNE)
    # cross-selective encoding + cross-attentive encoding
class CNE(NewsEncoder):
    def __init__(self, config: Config):
        super(CNE, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.word_embedding_dim = config.word_embedding_dim
        self.hidden_dim = config.hidden_dim
        self.news_embedding_dim = config.hidden_dim * 4 + config.category_embedding_dim + config.subCategory_embedding_dim
        # selective LSTM encoder
        self.title_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.content_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.title_H = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.title_M = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=True)
        self.content_H = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.content_M = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=True)
        # self-attention
        self.title_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        self.content_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        # cross-attention
        self.title_cross_attention = ScaledDotProduct_CandidateAttention(self.hidden_dim * 2, self.hidden_dim * 2, config.attention_dim)
        self.content_cross_attention = ScaledDotProduct_CandidateAttention(self.hidden_dim * 2, self.hidden_dim * 2, config.attention_dim)

    def initialize(self):
        super().initialize()
        for parameter in self.title_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        for parameter in self.content_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        nn.init.xavier_uniform_(self.title_H.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.title_M.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.title_M.bias)
        nn.init.xavier_uniform_(self.content_H.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.content_M.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.content_M.bias)
        self.title_self_attention.initialize()
        self.content_self_attention.initialize()
        self.title_cross_attention.initialize()
        self.content_cross_attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        title_mask = title_mask.view([batch_news_num, self.max_title_length])                                                                              # [batch_size * news_num, max_title_length]
        content_mask = content_mask.view([batch_news_num, self.max_content_length])                                                                        # [batch_size * news_num, max_content_length]
        title_mask[:, 0] = 1   # To avoid empty input of LSTM
        content_mask[:, 0] = 1 # To avoid empty input of LSTM
        title_length = title_mask.sum(dim=1, keepdim=False).long()                                                                                         # [batch_size * news_num]
        content_length = content_mask.sum(dim=1, keepdim=False).long()                                                                                     # [batch_size * news_num]
        sorted_title_length, sorted_title_indices = torch.sort(title_length, descending=True)                                                              # [batch_size * news_num]
        _, desorted_title_indices = torch.sort(sorted_title_indices, descending=False)                                                                     # [batch_size * news_num]
        sorted_content_length, sorted_content_indices = torch.sort(content_length, descending=True)                                                        # [batch_size * news_num]
        _, desorted_content_indices = torch.sort(sorted_content_indices, descending=False)                                                                 # [batch_size * news_num]
        # 1. word embedding
        title = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])                       # [batch_size * news_num, max_title_length, word_embedding_dim]
        content = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim])                 # [batch_size * news_num, max_content_length, word_embedding_dim]
        sorted_title = pack_padded_sequence(title.index_select(0, sorted_title_indices), sorted_title_length.cpu(), batch_first=True)                      # [batch_size * news_num, max_title_length, word_embedding_dim]
        sorted_content = pack_padded_sequence(content.index_select(0, sorted_content_indices), sorted_content_length.cpu(), batch_first=True)              # [batch_size * news_num, max_content_length, word_embedding_dim]
        # [Cross-selective Encoding]
        # 2. selective LSTM encoding
        # parallel bidirectional LSTMs (1),(2)
        # h: hidden state, c: cell state
        sorted_title_h, (sorted_title_h_n, sorted_title_c_n) = self.title_lstm(sorted_title)
        sorted_content_h, (sorted_content_h_n, sorted_content_c_n) = self.content_lstm(sorted_content)
        # semantic memory vector (3)
        sorted_title_m = torch.cat([sorted_title_c_n[0], sorted_title_c_n[1]], dim=1)                                                                      # [batch_size * news_num, hidden_dim * 2]
        sorted_content_m = torch.cat([sorted_content_c_n[0], sorted_content_c_n[1]], dim=1)                                                                # [batch_size * news_num, hidden_dim * 2]
        sorted_title_h, _ = pad_packed_sequence(sorted_title_h, batch_first=True, total_length=self.max_title_length)                                      # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_h, _ = pad_packed_sequence(sorted_content_h, batch_first=True, total_length=self.max_content_length)                                # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # sigmoid gate function (4),(5)
        sorted_title_gate = torch.sigmoid(self.title_H(sorted_title_h) + self.title_M(sorted_content_m).unsqueeze(dim=1))                                  # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_gate = torch.sigmoid(self.content_H(sorted_content_h) + self.content_M(sorted_title_m).unsqueeze(dim=1))                            # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # cross selective feature (final output of cross-selective encoding)
        title_h = (sorted_title_h * sorted_title_gate).index_select(0, desorted_title_indices)                                                             # [batch_size * news_num, max_title_length, hidden_dim * 2]
        content_h = (sorted_content_h * sorted_content_gate).index_select(0, desorted_content_indices)                                                     # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # [Cross-attentive Encoding]
        # 3. self-attention (6)
        title_self = self.title_self_attention(title_h, title_mask)                                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_self = self.content_self_attention(content_h, content_mask)                                                                                # [batch_size * news_num, hidden_dim * 2]
        # 4. cross-attention (7),(8)
        title_cross = self.title_cross_attention(title_h, content_self, title_mask)                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_cross = self.content_cross_attention(content_h, title_self, content_mask)                                                                  # [batch_size * news_num, hidden_dim * 2]
        news_representation = torch.cat([title_self + title_cross, content_self + content_cross], dim=1).view([batch_size, news_num, self.hidden_dim * 4]) # [batch_size, news_num, hidden_dim * 4]
        # 5. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                                              # [batch_size, news_num, news_embedding_dim]
        return news_representation


class CNN(NewsEncoder):
    def __init__(self, config: Config):
        super(CNN, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. CNN encoding
        c = self.dropout_(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))                                                           # [batch_size * news_num, max_sentence_length, cnn_kernel_num]
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.cnn_kernel_num])                        # [batch_size, news_num, cnn_kernel_num]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation


class MHSA(NewsEncoder):
    def __init__(self, config: Config):
        super(MHSA, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.feature_dim = config.head_num * config.head_dim
        self.multiheadAttention = MultiHeadAttention(config.head_num, config.word_embedding_dim, config.max_title_length, config.max_title_length, config.head_dim, config.head_dim)
        self.attention = Attention(config.head_num*config.head_dim, config.attention_dim)
        self.news_embedding_dim = config.head_num * config.head_dim + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        self.multiheadAttention.initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. multi-head self-attention
        c = self.dropout(self.multiheadAttention(w, w, w, mask))                                                                    # [batch_size * news_num, max_sentence_length, news_embedding_dim]
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.feature_dim])                           # [batch_size, news_num, news_embedding_dim]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation


class KCNN(NewsEncoder):
    def __init__(self, config: Config):
        super(KCNN, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.entity_embedding_dim = config.entity_embedding_dim
        self.context_embedding_dim = config.context_embedding_dim
        self.entity_embedding = nn.Embedding(num_embeddings=config.entity_size, embedding_dim=self.entity_embedding_dim)
        self.context_embedding = nn.Embedding(num_embeddings=config.entity_size, embedding_dim=self.context_embedding_dim)
        with open('entity_embedding-%s.pkl' % config.dataset, 'rb') as entity_embedding_f:
            self.entity_embedding.weight.data.copy_(pickle.load(entity_embedding_f))
        with open('context_embedding-%s.pkl' % config.dataset, 'rb') as context_embedding_f:
            self.context_embedding.weight.data.copy_(pickle.load(context_embedding_f))
        self.M_entity = nn.Linear(self.entity_embedding_dim, self.word_embedding_dim, bias=True)
        self.M_context = nn.Linear(self.context_embedding_dim, self.word_embedding_dim, bias=True)
        self.knowledge_cnn = Conv2D_Pool(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size, 3)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.M_entity.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.M_entity.bias)
        nn.init.xavier_uniform_(self.M_context.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.M_context.bias)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word & entity & context embedding
        word_embedding = self.word_embedding(title_text).view([batch_news_num, self.max_title_length, self.word_embedding_dim])                                  # [batch_size * news_num, max_title_length, word_embedding_dim]
        entity_embedding = self.entity_embedding(title_entity).view([batch_news_num, self.max_title_length, self.entity_embedding_dim])                          # [batch_size * news_num, max_title_length, entity_embedding_dim]
        context_embedding = self.context_embedding(title_entity).view([batch_news_num, self.max_title_length, self.context_embedding_dim])                       # [batch_size * news_num, max_title_length, context_embedding_dim]
        W = torch.stack([word_embedding, torch.tanh(self.M_entity(entity_embedding)), torch.tanh(self.M_context(context_embedding))], dim=3).permute(0, 2, 1, 3) # [batch_size * news_num, word_embedding_dim, max_title_length, 3]
        # 2. knowledge-aware CNN
        news_representation = self.knowledge_cnn(W).view([batch_size, news_num, self.cnn_kernel_num])                                                            # [batch_size, news_num, cnn_kernel_num]
        # 3. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                                                    # [batch_size, news_num, news_embedding_dim]
        return news_representation


class NAML(NewsEncoder):
    def __init__(self, config: Config):
        super(NAML, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.news_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
        self.subCategory_affine = nn.Linear(config.subCategory_embedding_dim, config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)

    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.subCategory_affine.weight)
        nn.init.zeros_(self.subCategory_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word embedding
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])       # [batch_size * news_num, max_title_length, word_embedding_dim]
        content_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim]) # [batch_size * news_num, max_content_length, word_embedding_dim]
        # 2. CNN encoding
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))                                                  # [batch_size * news_num, max_title_length, cnn_kernel_num]
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))                                            # [batch_size * news_num, max_content_length, cnn_kernel_num]
        # 3. attention layer
        title_representation = self.title_attention(title_c).view([batch_size, news_num, self.cnn_kernel_num])                               # [batch_size, news_num, cnn_kernel_num]
        content_representation = self.content_attention(content_c).view([batch_size, news_num, self.cnn_kernel_num])                         # [batch_size, news_num, cnn_kernel_num]
        # 4. category and subCategory encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)                              # [batch_size, news_num, cnn_kernel_num]
        subCategory_representation = F.relu(self.subCategory_affine(self.subCategory_embedding(subCategory)), inplace=True)                  # [batch_size, news_num, cnn_kernel_num]
        # 5. multi-view attention
        feature = torch.stack([title_representation, content_representation, category_representation, subCategory_representation], dim=2)    # [batch_size, news_num, 4, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                            # [batch_size, news_num, 4, 1]
        news_representation = (feature * alpha).sum(dim=2, keepdim=False)                                                                    # [batch_size, news_num, cnn_kernel_num]
        return news_representation
 
# PNE(Personalized News Encoder): NPA - news encoder
# - module 1: word embedding
# - module 2: convolutional neural network(CNN)
# - module 3: word-level personalized attention network

class PNE(NewsEncoder):
    def __init__(self, config: Config):
        super(PNE, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.personalized_embedding_dim = config.personalized_embedding_dim
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.dense = nn.Linear(config.user_embedding_dim, config.personalized_embedding_dim, bias=True)
        self.personalizedAttention = CandidateAttention(config.cnn_kernel_num, config.personalized_embedding_dim, config.attention_dim)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.dense.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense.bias)
        self.personalizedAttention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. CNN encoding
        c = self.dropout_(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))                                                           # [batch_size * news_num, max_sentence_length, cnn_kernel_num]
        # 3. attention layer
        q_w = F.relu(self.dense(user_embedding), inplace=True).repeat([news_num, 1])                                                # [batch_size * news_num, personalized_embedding_dim]
        news_representation = self.personalizedAttention(c, q_w, mask).view([batch_size, news_num, self.cnn_kernel_num])            # [batch_size, news_num, cnn_kernel_num]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation


class DAE(NewsEncoder):
    def __init__(self, config: Config):
        super(DAE, self).__init__(config)
        self.Alpha = config.Alpha
        assert self.Alpha > 0, 'Reconstruction loss weight must be greater than 0'
        self.f1 = nn.Linear(config.word_embedding_dim, config.hidden_dim, bias=True)
        self.f2 = nn.Linear(config.hidden_dim, config.word_embedding_dim, bias=True)
        self.news_embedding_dim = config.hidden_dim + config.category_embedding_dim + config.subCategory_embedding_dim
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.f1.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.f1.bias)
        nn.init.xavier_uniform_(self.f2.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.f2.bias)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        title_mask = title_mask.unsqueeze(dim=3)
        content_mask = content_mask.unsqueeze(dim=3)
        word_embedding = torch.sigmoid(((self.word_embedding(title_text) * title_mask).sum(dim=2) + (self.word_embedding(content_text) * content_mask).sum(dim=2)) \
                         / (title_mask.sum(dim=2, keepdim=False) + content_mask.sum(dim=2, keepdim=False)))           # [batch_size, news_num, word_embedding_dim]
        corrupted_word_embedding = self.dropout_(word_embedding)                                                      # [batch_size, news_num, word_embedding_dim]
        news_representation = torch.sigmoid(self.f1(corrupted_word_embedding))                                        # [batch_size, news_num, news_embedding_dim]
        denoised_word_embedding = torch.sigmoid(self.f2(news_representation))                                         # [batch_size, news_num, word_embedding_dim]
        self.auxiliary_loss = torch.norm(word_embedding - denoised_word_embedding, dim=2, keepdim=False) * self.Alpha # [batch_size, news_num]
        # feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                         # [batch_size, news_num, news_embedding_dim]
        return news_representation


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# class KnowledgePreparing(nn.Module):
#     def __init__(self, vocab_size: int, embed_size: int, knowledge_indices, alpha: float, beta: float, dropout: float = 0.3):
#         super().__init__()
        
#         common_entity_list = []
#         with open('./pre-trained/entities_yago.dict') as common_file:
#             while (line := common_file.readline().rstrip()):
#                 common_entity_list.append(line.split()[1].split('_')[0].lower())
        
#         def yield_tokens(data_iter, tokenizer):
#             for _, title, text in data_iter:
#                 yield tokenizer(title)
#                 yield tokenizer(str(text))
        
#         # build vocab
#         tokenizer = get_tokenizer('basic_english')
#         vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=['<unk>', '<sep>'])
#         vocab.set_default_index(vocab['<unk>'])
        
#         common_lookup_indices = vocab.lookup_indices(common_entity_list)
#         knowledge_indices['common'] = common_lookup_indices

# class KnowledgeEncoding(nn.Module):
    
#     def __init__(self, vocab_size: int, embed_size: int, knowledge_indices, alpha: float, dropout: float = 0.3):
#         super().__init__()

#         self.alpha = alpha

#         common_knowledge_path = './pre-trained/YAGO.RotatE.'

#         if embed_size == 128:  
#             common_knowledge_path += '128/entity_embedding.npy'
#         elif embed_size == 256:
#             common_knowledge_path += '256/entity_embedding.npy'
#         elif embed_size == 512:
#             common_knowledge_path += '512/entity_embedding.npy'
#         elif embed_size == 1024:
#             common_knowledge_path += '1024/entity_embedding.npy'
#         else:
#             print ('Wrong embedding dimension! Dimension should be 128, 256, 512, or 1024')
#             sys.exit(1)

#         common_pre_trained = np.load(common_knowledge_path)
#         common_knowledge = []
#         print('  - Reading Pre-trained Knowledge Embeddings...')
#         for idx in range(vocab_size):
#             mapping = 0
#             for j, vocab_idx in enumerate(knowledge_indices['common']):
#                 if idx != 0 and idx == vocab_idx:
#                     common_knowledge.append(common_pre_trained[j])
#                     mapping = 1
#                     break
#             if mapping == 0:
#                 common_knowledge.append(np.zeros(embed_size))
#             mapping = 0

#         self.common_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(common_knowledge))

#         self.fuse_knowledge_fc = nn.Linear(embed_size*2, embed_size)
#         self.dropout = nn.Dropout(p=dropout)
#         self.init_weights()

#     def init_weights(self) -> None:
#         initrange = 0.5
#         self.fuse_knowledge_fc.weight.data.uniform_(-initrange, initrange)
#         self.fuse_knowledge_fc.bias.data.zero_()

#     def forward(self, word_embeddings: Tensor, texts: Tensor) -> Tensor:

#         emb_with_ckwldg = (word_embeddings * self.alpha) + (self.common_knowledge(texts) * (1-self.alpha))
#         return self.dropout(emb_with_ckwldg)
