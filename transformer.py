import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
class SelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(SelfAttention, self).__init__()
        assert model_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.out_linear = nn.Linear(model_dim, model_dim)
        self.scale = self.head_dim ** 0.5  # 缩放因子

    def forward(self, x, mask=None):
        batch_size, seq_len, model_dim = x.size()
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)

        return self.out_linear(attn_output),attn_weights  # 返回注意力权重

class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))



class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, model_dim).to(device)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(np.log(10000.0) / model_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = SelfAttention(model_dim, num_heads)
        self.ffn = FeedForward(model_dim, ff_dim)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention
        attn_output, attn_weights = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        # Feed-Forward Network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.layer_norm2(x)

        return x,attn_weights

class MultiChannelTransformerModel3(nn.Module):
    def __init__(self, aa_input_dim, struct_input_dim,rsa_input_dim,model_dim, num_heads, num_layers, ff_dim, num_classes, max_seq_len):
        super(MultiChannelTransformerModel3, self).__init__()

        self.aa_embedding = nn.Embedding(aa_input_dim, model_dim, padding_idx=0)
        self.struct_embedding = nn.Embedding(struct_input_dim, model_dim,padding_idx=0)
        self.rsa_linear = nn.Linear(1, model_dim)  # 映射到 model_dim
        self.pos_encoding = PositionalEncoding(model_dim, max_seq_len)

        self.aa_encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

        self.struct_encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.rsa_encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

        self.fc1 = nn.Linear(model_dim*3, 128)
        self.fc2 = nn.Linear(128,num_classes)


    def forward(self, aa_sequences,struct_sequences,rsa_sequences,attention_mask=None):

        aa_embedding = self.aa_embedding(aa_sequences)
        aa_embedding = self.pos_encoding(aa_embedding)

        struct_embedding = self.struct_embedding(struct_sequences)
        struct_embedding = self.pos_encoding(struct_embedding)

        rsa_embedding = self.rsa_linear(rsa_sequences.unsqueeze(-1))
        rsa_embedding = self.pos_encoding(rsa_embedding)

        aa_attentions = []
        struct_attentions = []
        rsa_attentions = []

        aa_layer =  self.aa_encoder_layers[-1]
        aa_embedding,aa_attn_weights = aa_layer(aa_embedding,attention_mask)
        aa_attentions.append(aa_attn_weights.detach())

        struct_layer = self.struct_encoder_layers[-1]
        sturct_embedding, struct_attn_weights = struct_layer(struct_embedding, attention_mask)
        struct_attentions.append(struct_attn_weights.detach())

        rsa_layer = self.rsa_encoder_layers[-1]
        rsa_embedding, rsa_attn_weights = rsa_layer(rsa_embedding, attention_mask)
        rsa_attentions.append(rsa_attn_weights.detach())

        aa_rep = torch.mean(aa_embedding, dim=1)
        struct_rep = torch.mean(struct_embedding, dim=1)
        rsa_rep = torch.mean(rsa_embedding, dim=1)
        combined = torch.cat((aa_rep, struct_rep, rsa_rep), dim=1)
        output = self.fc2(F.relu(self.fc1(combined)))

        return output, {
            'aa_attentions': aa_attentions,
            'struct_attentions': struct_attentions,
            'rsa_attentions': rsa_attentions,
            'input_aa': aa_sequences,
            'input_struct': struct_sequences,
            'input_rsa': rsa_sequences
        }




class AECRLoss(nn.Module):
    def __init__(self, sigma=3, lambda_ent=1.0, lambda_loc=0.5):
        super().__init__()
        self.sigma = sigma
        self.lambda_ent = lambda_ent
        self.lambda_loc = lambda_loc
        self.gaussian_kernels = {}

    def _create_gaussian_kernel(self, L, device):
        i, j = torch.meshgrid(torch.arange(L, device=device),
                              torch.arange(L, device=device),
                              indexing='ij')
        kernel = torch.exp(-(i - j).float().pow(2) / (2 * self.sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, attentions_dict):
        total_loss = 0.0
        self.entropy = 0.0
        self.local = 0.0

        for channel_key in ['aa_attentions', 'struct_attentions', 'rsa_attentions']:
            layer_attentions = attentions_dict[channel_key]

            for layer_idx, attn_tensor in enumerate(layer_attentions):
                batch_size, num_heads, L, _ = attn_tensor.shape

                p = attn_tensor + 1e-10
                entropy = -p * torch.log(p)
                entropy_loss = entropy.mean()
                total_loss += self.lambda_ent * entropy_loss
                self.entropy += entropy_loss.item()

                if L not in self.gaussian_kernels:
                    self.gaussian_kernels[L] = self._create_gaussian_kernel(L, attn_tensor.device)
                gaussian_kernel = self.gaussian_kernels[L]

                local_similarity = torch.einsum('bhij,ij->bh', attn_tensor, gaussian_kernel)
                local_loss = 1 - local_similarity.mean()
                total_loss += self.lambda_loc * local_loss
                self.local += local_loss.item()

        return total_loss

