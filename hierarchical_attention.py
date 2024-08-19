import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, NamedTuple
import math

class HierarchicalMemory(NamedTuple):
    """Structure of the hierarchical memory.

    Where 'B' is batch size, 'M' is number of memories, 'C' is chunk size, and 'D'
    is memory dimension.
    """
    keys: torch.Tensor  # [B, M, D]
    contents: torch.Tensor  # [B, M, C, D]
    steps_since_last_write: torch.Tensor  # [B], steps since last memory write
    accumulator: torch.Tensor  # [B, C, D], accumulates experiences before write

def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    min_timescale: float = 2.,
    max_timescale: float = 1e4,
) -> torch.Tensor:
    """Creates sinusoidal encodings."""
    freqs = torch.arange(0, hidden_size, min_timescale)
    inv_freq = max_timescale ** (-freqs / hidden_size)
    pos_seq = torch.arange(sequence_length - 1, -1, -1.0)
    sinusoid_inp = pos_seq.unsqueeze(-1) @ inv_freq.unsqueeze(0)
    pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
    return pos_emb

class HierarchicalMemoryAttentionFinal(nn.Module):
    """Multi-head attention over hierarchical memory."""

    def __init__(self,
                 feature_size: int,
                 k: int,
                 num_heads: int = 1,
                 memory_position_encoding: bool = True,
                 init_scale: float = 2.) -> None:
        """Constructor.

        Args:
            feature_size: size of feature dimension of attention-over-memories
                embedding.
            k: number of memories to sample.
            num_heads: number of attention heads.
            memory_position_encoding: whether to add positional encodings to memories
                during within memory attention.
            init_scale: scale factor for Variance weight initializers.
        """
        super().__init__()
        self._size = feature_size
        self._k = k
        self._num_heads = num_heads
        self._memory_position_encoding = memory_position_encoding
        self._init_scale = init_scale

        self.query_proj = nn.Linear(feature_size, feature_size, bias=False)
        self.key_proj = nn.Linear(feature_size, feature_size, bias=False)
        self.value_proj = nn.Linear(feature_size, feature_size, bias=False)

        self.attention_layer = nn.MultiheadAttention(feature_size, num_heads, batch_first=True)

        # Initialize weights
        for layer in [self.query_proj, self.key_proj, self.value_proj]:
            nn.init.xavier_uniform_(layer.weight, gain=init_scale)

    def forward(self,
                queries: torch.Tensor,
                hm_memory: HierarchicalMemory,
                hm_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Do hierarchical attention over the stored memories."""
        batch_size, query_length, _ = queries.shape
        memory_batch_size, num_memories, memory_chunk_size, mem_embedding_size = hm_memory.contents.shape
        assert batch_size == memory_batch_size, "Batch sizes must match"

        # Project queries, keys, and values
        query_head = self.query_proj(queries)
        key_head = self.key_proj(hm_memory.keys.detach())
        value_head = self.value_proj(hm_memory.contents.detach())

        # Compute attention logits
        logits = torch.einsum("btd,bmd->btm", query_head, key_head)
        scaled_logits = logits / math.sqrt(self._size)

        # Apply mask if provided
        if (hm_mask is not None):
            masked_logits = torch.where(hm_mask, scaled_logits, torch.tensor(-1e9, device=scaled_logits.device))
        else:
            masked_logits = scaled_logits

        # Get top-k memories
        top_k_logits, top_k_indices = torch.topk(masked_logits, self._k, dim=-1)
        weights = F.softmax(top_k_logits, dim=-1)

        # Adjust dimensions for gather operation
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, memory_chunk_size, mem_embedding_size)

        # Gather top-k memory contents
        top_k_contents = torch.gather(hm_memory.contents.unsqueeze(1).expand(-1, query_length, -1, -1, -1).contiguous(), 2, top_k_indices_expanded)

        # Prepare memory contents and values for attention
        top_k_contents = top_k_contents.view(batch_size * query_length * self._k, memory_chunk_size, mem_embedding_size)

        # Apply positional encoding to memory contents
        if self._memory_position_encoding:
            position_embs = sinusoid_position_encoding(memory_chunk_size, mem_embedding_size).to(top_k_contents.device)
            top_k_contents = top_k_contents + position_embs.unsqueeze(0)

        # Prepare queries for attention
        queries_expanded = query_head.unsqueeze(2).expand(-1, -1, self._k, -1).contiguous()
        queries_expanded = queries_expanded.view(batch_size * query_length * self._k, 1, self._size)

        # Perform attention
        attn_output, _ = self.attention_layer(queries_expanded, top_k_contents, top_k_contents)

        # Reshape attention output
        attn_output = attn_output.view(batch_size, query_length, self._k, self._size)

        # Apply weights and sum
        weighted_output = weights.unsqueeze(-1) * attn_output
        output = weighted_output.sum(dim=2)

        return output

# --- Tensor Test ---
if __name__ == "__main__":
    # Define test parameters
    feature_size = 64
    k = 5
    num_heads = 4
    batch_size = 32
    query_length = 10
    num_memories = 100
    memory_chunk_size = 20

    # Instantiate the attention module
    hma_final = HierarchicalMemoryAttentionFinal(feature_size, k, num_heads)

    # Create random input tensors
    queries = torch.randn(batch_size, query_length, feature_size)
    hm_memory = HierarchicalMemory(
        keys=torch.randn(batch_size, num_memories, feature_size),
        contents=torch.randn(batch_size, num_memories, memory_chunk_size, feature_size),
        steps_since_last_write=torch.zeros(batch_size),
        accumulator=torch.zeros(batch_size, memory_chunk_size, feature_size)
    )
    hm_mask = torch.ones(batch_size, query_length, num_memories, dtype=torch.bool)

    # Perform the attention operation
    output_final = hma_final(queries, hm_memory, hm_mask)

    # Print the output shape
    output_shape_final = output_final.shape
    print("Output shape:", output_shape_final)

    # --- Additional Assertions ---
    output_shape_correct_final = output_shape_final == (batch_size, query_length, feature_size)
    contains_nan_final = torch.isnan(output_final).any().item()
    contains_inf_final = torch.isinf(output_final).any().item()

    print("Output shape correct:", output_shape_correct_final)
    print("Contains NaN values:", contains_nan_final)
    print("Contains Inf values:", contains_inf_final)
