import numpy as np
import torch
import torch.nn as nn

from geotransformer.modules.layers import build_dropout_layer


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        current_gpu_index = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory / (1024 ** 3)  # 显存总量(GB)
        used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)  # 已使用显存(GB)
        free_memory = total_memory - used_memory  # 剩余显存(GB)
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)



        """if input_shape[1] < 2000:
            omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        else:
            block_size = 1000
            n_blocks = input_shape[1]// block_size+1
            result_shape = (input_shape[1]*input_shape[2], self.div_term.shape[0], 1)
            omegas = torch.empty(result_shape)
            for i in range(n_blocks):
                start_idx = i * block_size
                end_idx = min((i + 1) * block_size, input_shape[1])
                emb_indices_block = emb_indices[0][start_idx:end_idx]
                emb_indices_block = emb_indices_block.view(-1, 1, 1)
                omegas_block = emb_indices_block * self.div_term.view(1, -1, 1)
                omegas[start_idx*input_shape[2]:end_idx*input_shape[2]] = omegas_block"""

        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        embeddings.to('cuda')
        print(embeddings.device)
        return embeddings


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout=None):
        super(LearnablePositionalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)  # (L, D)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = build_dropout_layer(dropout)

    def forward(self, emb_indices):
        r"""Learnable Positional Embedding.

        `emb_indices` are truncated to fit the finite embedding space.

        Args:
            emb_indices: torch.LongTensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        emb_indices = emb_indices.view(-1)
        max_emd_indices = torch.full_like(emb_indices, self.num_embeddings - 1)
        emb_indices = torch.minimum(emb_indices, max_emd_indices)
        embeddings = self.embeddings(emb_indices)  # (*, D)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.view(*input_shape, self.embedding_dim)
        return embeddings
