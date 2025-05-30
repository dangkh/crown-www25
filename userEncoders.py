import math
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, GCN
from torch.nn.utils.rnn import pack_padded_sequence
from layers import MultiHeadAttention, Attention, ScaledDotProduct_CandidateAttention, CandidateAttention, GCN_, Conv1D
from newsEncoders import NewsEncoder
# from torch_scatter import scatter_sum, scatter_softmax # need to be installed by following `https://pytorch-scatter.readthedocs.io/en/latest`
import numpy as np
import pickle

class UserEncoder(nn.Module):
    def __init__(self, news_encoder, config):
        super(UserEncoder, self).__init__()
        self.news_embedding_dim = news_encoder.news_embedding_dim
        self.news_encoder = news_encoder
        self.device = torch.device('cuda')
        self.auxiliary_loss = None
        self.word_embedding_dim = config.word_embedding_dim
        self.batch_size = config.batch_size

    # Input
    # user_title_text               : [batch_size, max_history_num, max_title_length]
    # user_title_mask               : [batch_size, max_history_num, max_title_length]
    # user_title_entity             : [batch_size, max_history_num, max_title_length]
    # user_content_text             : [batch_size, max_history_num, max_content_length]
    # user_content_mask             : [batch_size, max_history_num, max_content_length]
    # user_content_entity           : [batch_size, max_history_num, max_content_length]
    # category                      : [batch_size, news_num]
    # user_category                 : [batch_size, max_history_num]
    # user_subCategory              : [batch_size, max_history_num]
    # user_history_mask             : [batch_size, max_history_num]
    # user_history_graph            : [batch_size, max_history_num, max_history_num]
    # user_history_category_mask    : [batch_size, category_num]
    # user_history_category_indices : [batch_size, max_history_num]
    # user_embedding                : [batch_size, user_embedding]
    # candidate_news_representation : [batch_size, news_num, news_embedding_dim]
    # Output
    # user_representation           : [batch_size, news_embedding_dim]
    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, category, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        raise Exception('Function forward must be implemented at sub-class')

# Our proposed model: CROWN - user encoder
class CROWN(UserEncoder):
    def __init__(self, news_encoder, config):
        super(CROWN, self).__init__(news_encoder, config)
        
        self.attention_dim = config.attention_dim
        self.graph_sage = GraphSAGE(in_channels = self.news_embedding_dim,
                                    hidden_channels = self.news_embedding_dim,
                                    num_layers = 1,
                                    out_channels = self.news_embedding_dim,
                                    dropout = config.dropout_rate)

        self.user_node_embedding = nn.Parameter(torch.zeros([config.batch_size, self.news_embedding_dim]))
        

        self.K = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=False)
        self.Q = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=True)
        self.max_history_num = config.max_history_num
        self.attention_scalar = math.sqrt(float(self.attention_dim))
        self.affine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
    
    def initialize(self):
        nn.init.zeros_(self.user_node_embedding)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)
        nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine.bias)

    def create_bipartite_graph(self, user_history_mask, device):
        
        num_users, max_history_num = user_history_mask.size() 
        row_indices = torch.arange(num_users).view(-1, 1).repeat(1, max_history_num).view(-1).to(device) 
        col_indices = torch.arange(max_history_num).view(1, -1).repeat(num_users, 1).view(-1).to(device) 
        edge_index = torch.stack([row_indices, col_indices], dim=0) 
        
        return edge_index

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, category, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        batch_news_num = batch_size * news_num
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                  # [batch_size, max_history_num, news_embedding_dim]
        history_embedding = torch.cat([history_embedding, self.dropout_(self.user_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1))], dim=1)      # [batch_size, max_history_num + num_users, news_embedding_dim]

        # Create user-news bipartite graph
        edge_index = self.create_bipartite_graph(user_history_mask, history_embedding.device)
        # GNN convolution
        gcn_feature = self.graph_sage(history_embedding, edge_index)                            # [batch_size, max_history_num + num_users, news_embedding_dim]
        gcn_feature = gcn_feature[:, :self.max_history_num, :]                                  # [batch_size, max_history_num, news_embedding_dim]
        gcn_feature = gcn_feature.unsqueeze(dim=1).expand(-1, news_num, -1, -1)                 # [batch_size, news_num, max_history_num, news_embedding_dim]
        
        # Attention
        K = self.K(gcn_feature).view([batch_news_num, self.max_history_num, self.attention_dim])            # [batch_size * news_num, max_history_num, attention_dim]
        Q = self.Q(candidate_news_representation).view([batch_news_num, self.attention_dim, 1])             # [batch_size * news_num, attention_dim, 1]
        a = torch.bmm(K, Q).view([batch_news_num, self.max_history_num]) / self.attention_scalar            # [batch_size * news_num, max_history_num]
        alpha = F.softmax(a, dim=1)                                                                         # [batch_size * news_num, max_history_num]
        # bmm input: [batch_size * news_num, 1, max_history_num]
        # bmm mat2: [batch_size * news_num, max_history_num, news_embedding_dim]
        # bmm out: [batch_size * news_num, 1, news_embedding_dim]
        out = torch.bmm(alpha.unsqueeze(dim=1), gcn_feature.reshape([batch_news_num, self.max_history_num, self.news_embedding_dim]))       # [batch_size * news_num, 1, news_embedding_dim]
        out = out.squeeze(dim=1).view([batch_size, news_num, self.news_embedding_dim])                                                      # [batch_size, news_num, news_embedding_dim]
        
        user_representation = out                                                   # [batch_size, news_num, news_embedding_dim]

        return user_representation                                                          # [batch_size, news_num, news_embedding_dim]



# Structural User Encoding(SUE)
class SUE(UserEncoder):
    def __init__(self, news_encoder, config):
        super(SUE, self).__init__(news_encoder, config)
        self.attention_dim = max(config.attention_dim, self.news_embedding_dim // 4)
        self.proxy_node_embedding = nn.Parameter(torch.zeros([config.category_num, self.news_embedding_dim]))
        # Input
        # feature : [batch_size, node_num, feature_dim]
        # graph   : [batch_size, node_num, node_num]
        # Output
        # out     : [batch_size, node_num, feature_dim]
        self.gcn = GCN_(in_dim=self.news_embedding_dim, out_dim=self.news_embedding_dim, hidden_dim=self.news_embedding_dim, num_layers=config.gcn_layer_num, dropout=config.dropout_rate / 2, residual=not config.no_gcn_residual, layer_norm=config.gcn_layer_norm)
        self.intraCluster_K = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=False)
        self.intraCluster_Q = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=True)
        self.clusterFeatureAffine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.interClusterAttention = ScaledDotProduct_CandidateAttention(self.news_embedding_dim, self.news_embedding_dim, self.attention_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.category_num = config.category_num + 1 # extra one category index for padding news
        self.max_history_num = config.max_history_num
        self.attention_scalar = math.sqrt(float(self.attention_dim))

    def initialize(self):
        self.gcn.initialize()
        nn.init.zeros_(self.proxy_node_embedding)
        nn.init.xavier_uniform_(self.intraCluster_K.weight)
        nn.init.xavier_uniform_(self.intraCluster_Q.weight)
        nn.init.zeros_(self.intraCluster_Q.bias)
        nn.init.xavier_uniform_(self.clusterFeatureAffine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.clusterFeatureAffine.bias)
        self.interClusterAttention.initialize()


    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, category, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        batch_news_num = batch_size * news_num
        user_history_category_mask[:, -1] = 1
        user_history_category_mask = user_history_category_mask.unsqueeze(dim=1).expand(-1, news_num, -1).contiguous()                                  # [batch_size, news_num, category_num]
        user_history_category_indices = user_history_category_indices.unsqueeze(dim=1).expand(-1, news_num, -1)                                         # [batch_size, news_num, max_history_num]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                                                          # [batch_size, max_history_num, news_embedding_dim]
        # 1. GCN
        history_embedding = torch.cat([history_embedding, self.dropout_(self.proxy_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1))], dim=1) # [batch_size, max_history_num + category_num, news_embedding_dim]
        gcn_feature = self.gcn(history_embedding, user_history_graph) + history_embedding                                                               # [batch_size, max_history_num + category_num, news_embedding_dim]
        gcn_feature = gcn_feature[:, :self.max_history_num, :]                                                                                          # [batch_size, max_history_num, news_embedding_dim]
        gcn_feature = gcn_feature.unsqueeze(dim=1).expand(-1, news_num, -1, -1)                                                                         # [batch_size, news_num, max_history_num, news_embedding_dim]
        # 2. Intra-cluster attention
        K = self.intraCluster_K(gcn_feature).view([batch_news_num, self.max_history_num, self.attention_dim])                                           # [batch_size * news_num, max_history_num, attention_dim]
        Q = self.intraCluster_Q(candidate_news_representation).view([batch_news_num, self.attention_dim, 1])                                            # [batch_size * news_num, attention_dim, 1]
        a = torch.bmm(K, Q).view([batch_size, news_num, self.max_history_num]) / self.attention_scalar                                                  # [batch_size, news_num, max_history_num]
        # alpha_intra = scatter_softmax(a, user_history_category_indices, 2).unsqueeze(dim=3)                                                             # [batch_size, news_num, max_history_num, 1]
        # intra_cluster_feature = scatter_sum(alpha_intra * gcn_feature, user_history_category_indices, dim=2, dim_size=self.category_num)                # [batch_size, news_num, category_num, news_embedding_dim]
        intra_cluster_feature = intra_cluster_attention_sum(a, gcn_feature, user_history_category_indices, category_num=self.category_num)
        # perform nonlinear transformation on intra-cluster features
        intra_cluster_feature = self.dropout(F.relu(self.clusterFeatureAffine(intra_cluster_feature), inplace=True) + intra_cluster_feature)            # [batch_size, news_num, category_num, news_embedding_dim]
        # 3. Inter-cluster attention
        inter_cluster_feature = self.interClusterAttention(
            intra_cluster_feature.view([batch_news_num, self.category_num, self.news_embedding_dim]),
            candidate_news_representation.view([batch_news_num, self.news_embedding_dim]),
            mask=user_history_category_mask.view([batch_news_num, self.category_num])
        ).view([batch_size, news_num, self.news_embedding_dim])                                                                                         # [batch_size, news_num, news_embedding_dim]
        return inter_cluster_feature

def intra_cluster_attention_sum(a, gcn_feature, user_history_category_indices, category_num):
    """
    Args:
        a: Tensor of shape [B, N, H] – attention logits
        gcn_feature: Tensor of shape [B, N, H, D]
        user_history_category_indices: LongTensor of shape [B, N, H]
        category_num: int – total number of categories
        
    Returns:
        intra_cluster_feature: Tensor of shape [B, N, category_num, D]
    """
    B, N, H, D = gcn_feature.shape
    C = category_num

    alpha_intra = torch.zeros_like(a)

    for b in range(B):
        for n in range(N):
            idx = user_history_category_indices[b, n]  # (H,)
            scores = a[b, n]  # (H,)
            for c in range(C):
                mask = (idx == c)
                if mask.any():
                    scores_c = scores[mask]
                    softmax_c = F.softmax(scores_c, dim=0)
                    alpha_intra[b, n][mask] = softmax_c

    alpha_intra = alpha_intra.unsqueeze(dim=3)  # [B, N, H, 1]
    weighted_feature = alpha_intra * gcn_feature  # [B, N, H, D]

    intra_cluster_feature = torch.zeros(B, N, C, D, device=gcn_feature.device)

    for b in range(B):
        for n in range(N):
            idx = user_history_category_indices[b, n]  # (H,)
            feat = weighted_feature[b, n]  # (H, D)
            for c in range(C):
                mask = (idx == c).unsqueeze(-1)  # (H, 1)
                intra_cluster_feature[b, n, c] = (feat * mask.float()).sum(dim=0)

    return intra_cluster_feature
# LSTUR(Long Short-Term User Representations)
class LSTUR(UserEncoder):
    def __init__(self, news_encoder, config):
        super(LSTUR, self).__init__(news_encoder, config)
        self.masking_probability = 1.0 - config.long_term_masking_probability
        self.gru = nn.GRU(self.news_embedding_dim, self.news_embedding_dim, batch_first=True)

    def initialize(self):
        for parameter in self.gru.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, category, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        user_history_num = user_history_mask.sum(dim=1, keepdim=False).long()                                                                           # [batch_size]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                               user_category, user_subCategory, user_embedding)                                                          # [batch_size, max_history_num, news_embedding_dim]
        sorted_user_history_num, sorted_indices = torch.sort(user_history_num, descending=True)                                                         # [batch_size]
        _, desorted_indices = torch.sort(sorted_indices, descending=False)                                                                              # [batch_size]
        nonzero_indices = sorted_user_history_num.nonzero(as_tuple=False).squeeze(dim=1)
        if nonzero_indices.size(0) == 0:
            user_representation = user_embedding.unsqueeze(dim=1).expand(-1, news_num, -1)                                                              # [batch_size, news_num, news_embedding_dim]  
            return user_representation
        index = nonzero_indices[-1]
        if index + 1 == batch_size:
            sorted_user_embedding = user_embedding.index_select(0, sorted_indices)                                                                      # [batch_size, user_embedding_dim]
            if self.training and self.masking_probability != 1.0:
                sorted_user_embedding *= torch.bernoulli(torch.empty([batch_size, 1], device=self.device).fill_(self.masking_probability))              # [batch_size, user_embedding_dim]
            sorted_history_embedding = history_embedding.index_select(0, sorted_indices)                                                                # [batch_size, max_history_num, news_embedding_dim]
            packed_sorted_history_embedding = pack_padded_sequence(sorted_history_embedding, sorted_user_history_num.cpu(), batch_first=True)           # [batch_size, max_history_num, news_embedding_dim]
            _, h = self.gru(packed_sorted_history_embedding, sorted_user_embedding.unsqueeze(dim=0))                                                    # [1, batch_size, news_embedding_dim]
            user_representation = h.squeeze(dim=0).index_select(0, desorted_indices)                                                                    # [batch_size, news_embedding_dim]
        else:
            non_empty_indices = sorted_indices[:index+1]
            empty_indices = sorted_indices[index+1:]
            sorted_user_embedding = user_embedding.index_select(0, non_empty_indices)                                                                   # [batch_size, user_embedding_dim]
            if self.training and self.masking_probability != 1.0:
                sorted_user_embedding *= torch.bernoulli(torch.empty([index + 1, 1], device=self.device).fill_(self.masking_probability))               # [batch_size, user_embedding_dim]
            sorted_history_embedding = history_embedding.index_select(0, non_empty_indices)                                                             # [batch_size, max_history_num, news_embedding_dim]
            packed_sorted_history_embedding = pack_padded_sequence(sorted_history_embedding, sorted_user_history_num[:index+1].cpu(), batch_first=True) # [batch_size, max_history_num, news_embedding_dim]
            _, h = self.gru(packed_sorted_history_embedding, sorted_user_embedding.unsqueeze(dim=0))                                                    # [1, batch_size, news_embedding_dim]
            user_representation = torch.cat([h.squeeze(dim=0), user_embedding.index_select(0, empty_indices)], dim=0).index_select(0, desorted_indices) # [batch_size, news_embedding_dim]
        user_representation = user_representation.unsqueeze(dim=1).expand(-1, news_num, -1)                                                             # [batch_size, news_num, news_embedding_dim]
        return user_representation


class MHSA(UserEncoder):
    def __init__(self, news_encoder, config):
        super(MHSA, self).__init__(news_encoder, config)
        self.multiheadAttention = MultiHeadAttention(config.head_num, self.news_embedding_dim, config.max_history_num, config.max_history_num, config.head_dim, config.head_dim)
        self.affine = nn.Linear(config.head_num*config.head_dim, self.news_embedding_dim, bias=True)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        self.multiheadAttention.initialize()
        nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine.bias)
        self.attention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, category, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        news_num = candidate_news_representation.size(1)
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                  # [batch_size, max_history_num, news_embedding_dim]
        h = self.multiheadAttention(history_embedding, history_embedding, history_embedding, user_history_mask) # [batch_size, max_history_num, head_num * head_dim]
        h = F.relu(F.dropout(self.affine(h), training=self.training, inplace=True), inplace=True)               # [batch_size, max_history_num, news_embedding_dim]
        user_representation = self.attention(h).unsqueeze(dim=1).repeat(1, news_num, 1)                         # [batch_size, news_num, news_embedding_dim]
        return user_representation

# NPA - user encoder
class PUE(UserEncoder):
    def __init__(self, news_encoder, config):
        super(PUE, self).__init__(news_encoder, config)
        self.dense = nn.Linear(config.user_embedding_dim, config.personalized_embedding_dim, bias=True)
        self.personalizedAttention = CandidateAttention(self.news_embedding_dim, config.personalized_embedding_dim, config.attention_dim)

    def initialize(self):
        nn.init.xavier_uniform_(self.dense.weight, gain=nn.init.calculate_gain('relu')) #  for dense layer
        nn.init.zeros_(self.dense.bias) # for dense layer
        self.personalizedAttention.initialize() # for attention layer

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, category, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        news_num = candidate_news_representation.size(1)
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                                                # [batch_size, max_history_num, news_embedding_dim]
        q_d = F.relu(self.dense(user_embedding), inplace=True)                                                                                # [batch_size, personalized_embedding_dim]
        user_representation = self.personalizedAttention(history_embedding, q_d, user_history_mask).unsqueeze(dim=1).expand(-1, news_num, -1) # [batch_size, news_num, news_embedding_dim]
        return user_representation


class ATT(UserEncoder):
    def __init__(self, news_encoder, config):
        super(ATT, self).__init__(news_encoder, config)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        self.attention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, category, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        news_num = candidate_news_representation.size(1)
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)            # [batch_size, max_history_num, news_embedding_dim]
        user_representation = self.attention(history_embedding).unsqueeze(dim=1).expand(-1, news_num, -1) # [batch_size, news_embedding_dim]
        return user_representation


class CATT(UserEncoder):
    def __init__(self, news_encoder, config):
        super(CATT, self).__init__(news_encoder, config)
        self.affine1 = nn.Linear(self.news_embedding_dim * 2, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=True)
        self.max_history_num = config.max_history_num

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)
        nn.init.zeros_(self.affine2.bias)

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, category, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        news_num = candidate_news_representation.size(1)
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                                  # [batch_size, max_history_num, news_embedding_dim]
        user_history_mask = user_history_mask.unsqueeze(dim=1).expand(-1, news_num, -1)                                         # [batch_size, news_num, max_history_num]
        candidate_news_representation = candidate_news_representation.unsqueeze(dim=2).expand(-1, -1, self.max_history_num, -1) # [batch_size, news_num, max_history_num, news_embedding_dim]
        history_embedding = history_embedding.unsqueeze(dim=1).expand(-1, news_num, -1, -1)                                     # [batch_size, news_num, max_history_num, news_embedding_dim]
        concat_embeddings = torch.cat([candidate_news_representation, history_embedding], dim=3)                                # [batch_size, news_num, max_history_num, news_embedding_dim * 2]
        hidden = F.relu(self.affine1(concat_embeddings), inplace=True)                                                          # [batch_size, news_num, max_history_num, attention_dim]
        a = self.affine2(hidden).squeeze(dim=3)                                                                                 # [batch_size, news_num, max_history_num]
        alpha = F.softmax(a.masked_fill(user_history_mask == 0, -1e9), dim=2)                                                   # [batch_size, news_num, max_history_num]
        user_representation = (alpha.unsqueeze(dim=3) * history_embedding).sum(dim=2, keepdim=False)                            # [batch_size, news_num, news_embedding_dim]
        return user_representation


class FIM(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(FIM, self).__init__(news_encoder, config)
        # assert type(self.news_encoder) == HDC, 'For FIM, the news encoder must be HDC'
        self.HDC_sequence_length = news_encoder.HDC_sequence_length
        self.max_history_num = config.max_history_num
        self.scalar = math.sqrt(float(config.HDC_filter_num))
        self.conv_3D_a = nn.Conv3d(in_channels=4, out_channels=config.conv3D_filter_num_first, kernel_size=config.conv3D_kernel_size_first)
        self.conv_3D_b = nn.Conv3d(in_channels=config.conv3D_filter_num_first, out_channels=config.conv3D_filter_num_second, kernel_size=config.conv3D_kernel_size_second)
        self.maxpool_3D = torch.nn.MaxPool3d(kernel_size=config.maxpooling3D_size, stride=config.maxpooling3D_stride)

    def initialize(self):
        pass

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, category, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        candidate_news_d0, candidate_news_dL = candidate_news_representation
        history_embedding_d0, history_embedding_dL = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                                                       user_content_text, user_content_mask, user_content_entity, \
                                                                       user_category, user_subCategory, user_embedding)
        batch_size = candidate_news_d0.size(0)
        news_num = candidate_news_d0.size(1)
        batch_news_num = batch_size * news_num
        # 1. compute 3D matching images
        candidate_news_d0 = candidate_news_d0.unsqueeze(dim=2).permute(0, 1, 2, 4 ,3)                                                       # [batch_size, news_num, 1, HDC_sequence_length, HDC_filter_num]
        candidate_news_dL = candidate_news_dL.unsqueeze(dim=2).permute(0, 1, 2, 3 ,5, 4)                                                    # [batch_size, news_num, 1, 3, HDC_sequence_length, HDC_filter_num]
        history_embedding_d0 = history_embedding_d0.unsqueeze(dim=1)                                                                        # [batch_size, 1, max_history_num, HDC_filter_num, HDC_sequence_length]
        history_embedding_dL = history_embedding_dL.unsqueeze(dim=1)                                                                        # [batch_size, 1, max_history_num, 3, HDC_filter_num, HDC_sequence_length]
        matching_images_d0 = torch.matmul(candidate_news_d0, history_embedding_d0) / self.scalar                                            # [batch_size, news_num, max_history_num, HDC_sequence_length, HDC_sequence_length]
        matching_images_dL = torch.matmul(candidate_news_dL, history_embedding_dL) / self.scalar                                            # [batch_size, news_num, max_history_num, 3, HDC_sequence_length, HDC_sequence_length]
        matching_images = torch.cat([matching_images_d0.unsqueeze(dim=3), matching_images_dL], dim=3).permute(0, 1, 3, 2, 4, 5)             # [batch_size, news_num, 4, max_history_num, HDC_sequence_length, HDC_sequence_length]
        matching_images = matching_images.view(batch_news_num, 4, self.max_history_num, self.HDC_sequence_length, self.HDC_sequence_length) # [batch_size * news_num, 4, max_history_num, HDC_sequence_length, HDC_sequence_length]
        # 2. 3D convolution layers
        Q1 = F.elu(self.conv_3D_a(matching_images), inplace=True)                                                                           # [batch_size * news_num, conv3D_filter_num_first, max_history_num, HDC_sequence_length, HDC_sequence_length]
        Q1 = self.maxpool_3D(Q1)                                                                                                            # [batch_size * news_num, conv3D_filter_num_first, max_history_num_conv1_size, HDC_sequence_length_conv1_size, HDC_sequence_length_conv1_size]
        Q2 = F.elu(self.conv_3D_b(Q1), inplace=True)                                                                                        # [batch_size * news_num, conv3D_filter_num_second, max_history_num_pool1_size, HDC_sequence_length_pool1_size, HDC_sequence_length_pool1_size]
        Q2 = self.maxpool_3D(Q2)                                                                                                            # [batch_size * news_num, conv3D_filter_num_second, max_history_num_conv2_size, HDC_sequence_length_conv2_size, HDC_sequence_length_conv2_size]
        salient_signals = Q2.view([batch_size, news_num, -1])                                                                               # [batch_size * news_num, feature_size]
        return salient_signals


class GRU(UserEncoder):
    def __init__(self, news_encoder, config):
        super(GRU, self).__init__(news_encoder, config)
        self.gru = nn.GRU(self.news_embedding_dim, config.hidden_dim, batch_first=True)
        self.dec = nn.Linear(config.hidden_dim, self.news_embedding_dim, bias=True)

    def initialize(self):
        for parameter in self.gru.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        nn.init.xavier_uniform_(self.dec.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.dec.bias)

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, category, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        user_history_num = user_history_mask.sum(dim=1, keepdim=False).long()                                                                           # [batch_size]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)                                                          # [batch_size, max_history_num, news_embedding_dim]
        sorted_user_history_num, sorted_indices = torch.sort(user_history_num, descending=True)                                                         # [batch_size]
        _, desorted_indices = torch.sort(sorted_indices, descending=False)                                                                              # [batch_size]
        nonzero_indices = sorted_user_history_num.nonzero(as_tuple=False).squeeze(dim=1)
        if nonzero_indices.size(0) == 0:
            user_representation = torch.zeros([batch_size, news_num, self.news_embedding_dim], device=self.device)                                      # [batch_size, news_num, news_embedding_dim]
            return user_representation
        index = nonzero_indices[-1]
        if index + 1 == batch_size:
            sorted_history_embedding = history_embedding.index_select(0, sorted_indices)                                                                # [batch_size, max_history_num, news_embedding_dim]
            packed_sorted_history_embedding = pack_padded_sequence(sorted_history_embedding, sorted_user_history_num.cpu(), batch_first=True)           # [batch_size, max_history_num, news_embedding_dim]
            _, h = self.gru(packed_sorted_history_embedding)                                                                                            # [1, batch_size, news_embedding_dim]
            h = torch.tanh(self.dec(h.squeeze(dim=0)))                                                                                                  # [batch_size, news_embedding_dim]
            user_representation = h.index_select(0, desorted_indices)                                                                                   # [batch_size, news_embedding_dim]
        else:
            non_empty_indices = sorted_indices[:index+1]
            sorted_history_embedding = history_embedding.index_select(0, non_empty_indices)                                                             # [batch_size, max_history_num, news_embedding_dim]
            packed_sorted_history_embedding = pack_padded_sequence(sorted_history_embedding, sorted_user_history_num[:index+1].cpu(), batch_first=True) # [batch_size, max_history_num, news_embedding_dim]
            _, h = self.gru(packed_sorted_history_embedding)                                                                                            # [1, batch_size, news_embedding_dim]
            h = torch.tanh(self.dec(h.squeeze(dim=0)))                                                                                                  # [batch_size, news_embedding_dim]
            user_representation = torch.cat([h, torch.zeros([batch_size - 1 - index, self.news_embedding_dim], device=self.device)], \
                                            dim=0).index_select(0, desorted_indices)                                                                    # [batch_size, news_embedding_dim]
        user_representation = user_representation.unsqueeze(dim=1).expand(-1, news_num, -1)                                                             # [batch_size, news_num, news_embedding_dim]
        return user_representation
    
