"""The internal transformer module code.

Based on https://raw.githubusercontent.com/huggingface/transformers/e3cc4487fe66e03ec85970ea2db8e5fb34c455f4/src/transformers/models/gpt_neo/modeling_gpt_neo.py
"""  # noqa E501

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from ..data.data_embedding_layer import DataEmbeddingLayer, MeasIndexGroupOptions
from ..data.types import PytorchBatch
from .config import StructuredEventProcessingMode, StructuredTransformerConfig
from .model_output import TransformerOutputWithPast, RetreivalTransformerMiddleLayersOutput
from .structured_attention import StructuredAttention

logger = logging.get_logger(__name__)


def expand_mask(mask: torch.BoolTensor, dtype: torch.dtype) -> torch.Tensor:
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, 1, seq_len]` and converts to float.

    This enables broadcasting to [bsz, num_heads, from_seq_len, to_seq_len] by converting the size [bsz,
    seq_len] to [bsz, 1, 1, seq_len] and converts from a boolean form to an attention weights masking form,
    which has 0 where the original mask was True and the minimum possible floating point expressible value
    where it was False.

    Args:
        mask: The event presence/absence mask of shape `[bsz, seq_len]`.
        dtype: The target dtype of the attention mask.

    Returns:
        The passed event indicator mask reshaped and type converted, unless mask is `None` in which case
        returns `None`.

    Examples:
        >>> import torch
        >>> assert expand_mask(None, None) is None
        >>> mask = torch.BoolTensor([
        ...     [True, True, False, False],
        ...     [True, True, True, False],
        ... ])
        >>> dtype = torch.float16
        >>> print(expand_mask(mask, dtype))
        tensor([[[[    -0.,     -0., -65504., -65504.]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[    -0.,     -0.,     -0., -65504.]]]], dtype=torch.float16)
    """
    if mask is None:
        return None

    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    attention_mask = mask[:, None, None, :]

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_mask = attention_mask.to(dtype=dtype)  # fp16 compatibility
    attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min

    return attention_mask


class Attention(nn.Module):
    """Implements attention mechanism

    This involves performing the (self-)attention operation and returning the result along with
    some optional additional outputs. The constructor of this class accepts three arguments, which determine
    the configuration of the self-attention mechanism.

    Args:
        config: An instance of StructuredTransformerConfig which contains various
            configuration parameters.

    Raises:
        ValueError: If the product of `num_heads` and `head_dim` from the config
            does not match `embed_dim`.
    """

    def __init__(self, config: StructuredTransformerConfig):
        super().__init__()
        max_seq_len = config.max_seq_len
        bias = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool))
        bias = bias.view(1, 1, max_seq_len, max_seq_len)

        self.register_buffer("bias", bias)
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout_p = config.attention_dropout

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and "
                f"`num_heads`: {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Splits the last dimension of a tensor into `num_heads` and `attn_head_size`"""
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merges the last two dimensions of a tensor into a single dimension"""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """Performs the attention operation.

        Returns:
            A tuple containing the output of the attention operation and the attention weights.
        """
        if head_mask is not None:
            raise ValueError("layer_head_mask different than None is unsupported for now")

        batch_size = query.shape[0]

        mask_value = torch.finfo(value.dtype).min
        mask_value = torch.full([], mask_value, dtype=value.dtype)

        # in gpt-neo-x and gpt-j the query and keys are always in fp32
        # thus we need to cast them to the value dtype
        query = query.to(value.dtype)
        key = key.to(value.dtype)
        # query, key, and value are all of shape (batch, head, seq_length, head_features)

        if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
            raise ValueError(
                "BetterTransformer does not support padding='max_length' with a batch size of 1."
            )

        dropout_p = self.attn_dropout_p if self.training else 0.0
        is_batched_inference = batch_size > 1 and not self.training

        if not is_batched_inference:
            if query.shape[2] > 1:  # seqlen > 1
                sdpa_result = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
                )
            else:
                sdpa_result = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False
                )
        else:
            query_length, key_length = query.size(-2), key.size(-2)

            # causal_mask is always [True, ..., True] otherwise, so executing this
            # is unnecessary
            if query_length > 1:
                causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

                causal_mask = torch.where(causal_mask, 0, mask_value)

                # torch.Tensor.expand does no memory copy
                causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
                if attention_mask is not None:
                    attention_mask = causal_mask + attention_mask

            sdpa_result = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
            )

        # in gpt-neo-x and gpt-j the query and keys are always in fp32
        # thus we need to cast them to the value dtype
        sdpa_result = sdpa_result.to(value.dtype)

        return sdpa_result, None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        static_kv_first: bool = False,
        kv_hidden_states: torch.Tensor | None = None,
    ):
        """Applies the attention mechanism to the input hidden states.

        Args:
            static_kv_first: In the case of attention over the dependency graph, the history embedding is
                dropped after processing, so we want to only use it as a KV, not as a query.
            kv_hidden_states: If not None, compute cross attention considering `hidden_states` as the query
                and `kv_hidden_states` as the input to key and value networks.

        Returns:
            A tuple containing the output of the attention mechanism and a dictionary of optional outputs.
        """
        if kv_hidden_states is None:
            kv_hidden_states = hidden_states

        query = self.q_proj(hidden_states)
        key = self.k_proj(kv_hidden_states)
        value = self.v_proj(kv_hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        # query, key, and value are all of shape (batch, head, seq_length, head_features)

        if static_kv_first:
            # In this case, we are only interested in performing the attention update over the non-static
            # queries.
            query = query[:, :, 1:, :]

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = None
        if use_cache:
            present = (key, value)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)

        outputs = {"present_key_value": present}
        if output_attentions:
            outputs["attn_weights"] = attn_weights

        return attn_output, outputs  # a, {present, (attentions)}


class ChunkedCrossAttention(nn.Module):
    def __init__(self, *, hidden, chunk_len, num_heads, kv_dim=None):
        """Used in the decoder to pay attention to the retrieved neighbor chunks

        Learn more: https://arxiv.org/abs/2112.04426
        Implementation is based on https://nn.labml.ai/transformers/retro/model.html
        """
        if chunk_len is None:
            raise RuntimeError("chunk_len must be provided")

        super().__init__()
        self.hidden = hidden
        self.chunk_len = chunk_len
        self.num_heads = num_heads
        assert hidden % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_size = self.hidden // self.num_heads

        # TODO: support multiquery (shared-key) attention
        kv_dim = kv_dim or hidden
        self.k_proj = nn.Linear(kv_dim, self.hidden)
        self.v_proj = nn.Linear(kv_dim, self.hidden)
        self.q_proj = nn.Linear(self.hidden, self.hidden)
        self.out_proj = nn.Linear(self.hidden, self.hidden)

    def forward(self, query, kv):
        """
        Args:
            q: torch.FloatTensor[batch_size, seq_len, hidden]
            kv: torch.FloatTensor[batch_size, n_chunks, neighbors, neighbor_len, kv_dim]
        """
        batch_size, n_chunks, n_neighbors, neighbor_len, _ = kv.shape
        q_batch_size, seq_len, q_hidden = query.shape
        assert q_batch_size == batch_size, f"q_batch_size: {q_batch_size}, batch_size: {batch_size}"
        assert q_hidden == self.hidden, f"q_hidden: {q_hidden}, hidden: {self.hidden}"

        if n_chunks == 0:
            # No attention if there are no chunks (for short inputs when sampling)
            return query

        # TODO: add positional information to q and kv

        # Remove the first chunk_len - 1 embeddings.
        # The input pays attention to neighbors retrieved and encoded using the past tokens only;
        # so that there is no information leakage.
        # That is the retrieved neighbors from the first chunks will have information from the first chunk.
        # So by shifting the sequence to the left by chunk_len - 1 we make sure that information only flows to the right.
        query = query[:, self.chunk_len - 1:]

        if query.shape[1] > n_chunks * self.chunk_len:
            raise RuntimeError(
                f"Chunked cross attention expects q.shape[1] <= n_chunks * chunk_len, "
                f"got q.shape[1]: {query.shape[1]}, n_chunks: {n_chunks}, chunk_len: {self.chunk_len}. "
                f"It's likely you're not forming retreived_hidden_states correctly"
                f"They should be of shape (batch_size, n_chunks, n_neighbors, neighbor_len, hidden_size)"
                f"And the retreival query used to retreive the hiddens should be of shape (batch_size, n_chunks, chunk_len, hidden_size)"
            )

        if query.shape[1] < n_chunks * self.chunk_len:
            # Append empty embeddings to the end to be able to split the input into chunks
            # NOTE: this code a bit sus, need to figure out if we assume that seq_len == n_chunks * chunk_len
            add_len = n_chunks * self.chunk_len - query.shape[1]
            assert 0 <= add_len < self.chunk_len, f"add_len: {add_len}, chunk_len: {self.chunk_len}"
            query = torch.cat((query, query.new_zeros(batch_size, add_len, q_hidden)), dim=1)

        query = query.reshape(batch_size, n_chunks, self.chunk_len, q_hidden)

        query = self.q_proj(query).view(batch_size, n_chunks, self.chunk_len, self.num_heads, self.head_size)
        k = self.k_proj(kv).view(batch_size, n_chunks, n_neighbors, neighbor_len, self.num_heads, self.head_size)
        v = self.v_proj(kv).view(batch_size, n_chunks, n_neighbors, neighbor_len, self.num_heads, self.head_size)

        attn = torch.einsum('bcihd,bcnjhd->bchinj', query, k)  # (batch_size, n_chunks, n_neighbors, neighbor_len, n_neighbors)
        assert attn.shape == (batch_size, n_chunks, self.num_heads, self.chunk_len, n_neighbors, neighbor_len)

        attn = attn / math.sqrt(self.head_size)

        # Weird part: Apply softmax over the last two dimensions neighbors * neighbor_len
        # Each query from the chunk attends to text from all neighbors concatenated together
        attn = F.softmax(attn.view(*attn.shape[:-2], -1)).view(attn.shape)

        h = torch.einsum("bchinj,bcnjhd->bcihd", attn, v)
        h = h.reshape(batch_size, n_chunks * self.chunk_len, q_hidden)

        h = torch.cat((h.new_zeros(batch_size, self.chunk_len - 1, q_hidden), h), dim=1)
        h = h[:, :seq_len]

        # TODO: use F.scaled_dot_product_attention instead of torch.einsum
        # Expected input shapes
        # query: (batch_size, ..., q_seq_len, hidden).
        # key: (batch_size, ..., kv_seq_len, hidden).
        # value: (batch_size, ..., kv_seq_len, v_hidden).
        # dropout_p = self.attn_dropout_p if self.training else 0.0
        # attn = F.scaled_dot_product_attention(
        #     query=q, key=k, value=v, attn_mask=None, dropout_p=dropout_p, is_causal=False
        # )
        # output: (batch_size, ..., q_seq_len, v_hidden)

        return h


class MLP(nn.Module):
    """Applies a multilayer perceptron (MLP) to the `hidden_states`"""

    def __init__(self, config: StructuredTransformerConfig):
        super().__init__()
        embed_dim = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * embed_dim

        self.in_proj = nn.Linear(embed_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    """An inner block in a transformer architecture that consists of attention and MLP layers.

    Args:
        config: Configuration parameters for the structured transformer.
        layer_id: Unique identifier for the layer.
        is_seq: Flag indicating whether the block is sequential.
    """
    def __init__(
        self,
        config: StructuredTransformerConfig,
        *,
        layer_id: int,
        is_seq: bool,
        is_retreival_block: bool = False,
    ):
        super().__init__()
        self.is_retreival_block = is_retreival_block
        self.layer_id = layer_id
        self.is_seq = is_seq

        self.pre_attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(config)

        self.cross_attn = None
        self.cross_attn_ln = None
        if is_retreival_block:
            self.cross_attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.cross_attn = ChunkedCrossAttention(
                chunk_len=config.chunked_cross_attention_chunk_len,
                num_heads=config.num_attention_heads,
                hidden=config.hidden_size,
            )

        self.pre_mlp_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        static_kv_first: bool = False,
        retreived_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Conducts the forward pass for the inner block.

        Args:
            hidden_states: Input tensor of shape `(batch_size, seq_len, hidden_size)`.
            layer_past: Cache of past hidden states for more efficient decoding.
            use_cache: Whether to use caching.
            output_attentions: Whether to return attention probabilities in the output.
            static_kv_first: Whether the static key-value pair comes first.
            retreived_hidden_states: Hidden states to condition on
                Tensor of shape `(batch_size, n_chunks, n_neighbors, neighbor_len, hidden_size)`.

        Returns:
            tuple: Modified hidden states and a dictionary containing present key-value pair and
            attention weights (if `output_attentions=True`).
        """

        # If we have a static kv entry first, we don't want to process it in the rest of the block, so we drop
        # it from the residual.
        residual = hidden_states if not static_kv_first else hidden_states[:, 1:, :]

        hidden_states = self.pre_attn_ln(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            static_kv_first=static_kv_first,
        )
        attn_output, outputs = attn_outputs  # output_attn: a, {present, (attentions)}

        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.pre_mlp_ln(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if self.cross_attn is not None:
            if retreived_hidden_states is None:
                raise ValueError("retreived_hidden_states must be provided for chunked cross attention (second half of the model layers)")
            # TODO: support caching
            residual = hidden_states
            hidden_states = self.cross_attn_ln(hidden_states)
            hidden_states = self.cross_attn(query=hidden_states, kv=retreived_hidden_states)
            hidden_states = residual + hidden_states

        if not use_cache:
            outputs.pop("present_key_value")
        return hidden_states, outputs


class StructuredTransformerBlock(nn.Module):
    """A block for structured attention with both sequential and dependency graph modules.

    Args:
        config: Configuration parameters for the structured transformer.
        layer_id: Unique identifier (depth index) for the layer.
    """

    def __init__(self, config: StructuredTransformerConfig, layer_id: int):
        super().__init__()

        if config.do_full_block_in_seq_attention:
            seq_block = TransformerBlock(config, layer_id, is_seq=True)
        else:
            seq_block = nn.Sequential(
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon),
                Attention(config)
            )

        if config.do_full_block_in_dep_graph_attention:
            dep_graph_block = TransformerBlock(config, layer_id, is_seq=False)
        else:
            dep_graph_block = nn.Sequential(
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon),
                Attention(config)
            )

        self.block = StructuredAttention(
            seq_module=seq_block,
            dep_graph_module=dep_graph_block,
        )

    def forward(
        self, *args, **kwargs
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor | None] | None]]:
        """Conducts the forward pass for the structured transformer block.

        Args:
            args: Variable length argument list.
            kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: Modified input tensor and a dictionary containing present key-value pair and
            attention weights.
        """

        return self.block(*args, **kwargs)


class StructuredTransformerPreTrainedModel(PreTrainedModel):
    """The base pre-trained model class for Transformer models."""

    config_class = StructuredTransformerConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["StructuredTransformerBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, StructuredTransformerPreTrainedModel):
            module.gradient_checkpointing = value


def time_from_deltas(batch: PytorchBatch) -> torch.Tensor:
    """Given a batch of time deltas, compute the relative time-since-start for each event.

    Args:
        batch: The input batch

    Examples:
        >>> batch = PytorchBatch(
        ...     event_mask=torch.BoolTensor([
        ...         [True, True, True], [True, True, False]
        ...     ]),
        ...     time_delta=torch.Tensor([[1.0, 3.2, 0.0], [1.4, 0.0, 1.0]])
        ... )
        >>> print(time_from_deltas(batch))
        tensor([[0.0000, 1.0000, 4.2000],
                [0.0000, 1.4000, 1.4000]])
    """
    t_deltas = batch["time_delta"]

    if batch.event_mask is not None:
        t_deltas = torch.where(batch.event_mask, t_deltas, torch.zeros_like(t_deltas))

    return torch.hstack([torch.zeros_like(t_deltas[:, :1]), t_deltas.cumsum(-1)[:, :-1]])


class LearnableFrequencySinusoidalTemporalPositionEncoding(torch.nn.Module):
    """A module for applying time-based position encodings to a PytorchBatch.

    Adapted from :footcite:t:`wang2021on` (`link`_).

    .. _link: https://openreview.net/pdf?id=onxoVA9FxMw

    .. footbibliography::

    Args:
        embedding_dim: The desired size of the output embedding. Unlike many position embedding
            implementations, this does not need to be even.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_timepoint: float = 10000.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        # div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(max_timepoint) / embedding_dim))

        size = math.ceil(embedding_dim / 2)
        div_term = torch.empty(
            size,
        )
        torch.nn.init.normal_(div_term)

        # We still want this to work for odd embedding dimensions, so we'll lop off the end of the cos
        # embedding. This is not a principled decision, but enabling odd embedding dimensions helps avoid edge
        # cases during hyperparameter tuning when searching over possible embedding spaces.
        if self.embedding_dim % 2 == 0:
            self.sin_div_term = torch.nn.Parameter(div_term, requires_grad=True)
            self.cos_div_term = torch.nn.Parameter(div_term, requires_grad=True)
        else:
            self.sin_div_term = torch.nn.Parameter(div_term, requires_grad=True)
            self.cos_div_term = torch.nn.Parameter(div_term[:-1], requires_grad=True)

    def forward(self, batch: PytorchBatch) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: The input batch to process.

        Returns:
            The temporal position embeddings tensor of shape (bsz, seq_len)
        """

        t = time_from_deltas(batch) if batch.get("time", None) is None else batch["time"]
        bsz, seq_len = t.shape
        device = t.device

        # First, we go from deltas to time values and unsqueeze it for broadcasting through the hidden dim.
        t = t.unsqueeze(-1)

        # temporal_embeddings will be our output container.
        # It should have shape (batch size, sequence length, embedding dim), and be on the same device as the
        # timepoints.
        temporal_embeddings = torch.zeros(bsz, seq_len, self.embedding_dim, device=device)

        temporal_embeddings[:, :, 0::2] = torch.sin(t * self.sin_div_term.unsqueeze(0).unsqueeze(0))
        temporal_embeddings[:, :, 1::2] = torch.cos(t * self.cos_div_term.unsqueeze(0).unsqueeze(0))

        return temporal_embeddings


class TemporalPositionEncoding(torch.nn.Module):
    """A module for applying time-based position encodings to a PytorchBatch.

    Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:
        embedding_dim: The desired size of the output embedding. Unlike many position embedding
            implementations, this does not need to be even.
        max_timepoint: The maximum observed timepoint, used to initialize the frequency space.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_timepoint: float = 10000.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(max_timepoint) / embedding_dim))

        # We still want this to work for odd embedding dimensions, so we'll lop off the end of the cos
        # embedding. This is not a principled decision, but enabling odd embedding dimensions helps avoid edge
        # cases during hyperparameter tuning when searching over possible embedding spaces.
        if self.embedding_dim % 2 == 0:
            self.sin_div_term = torch.nn.Parameter(div_term, requires_grad=False)
            self.cos_div_term = torch.nn.Parameter(div_term, requires_grad=False)
        else:
            self.sin_div_term = torch.nn.Parameter(div_term, requires_grad=False)
            self.cos_div_term = torch.nn.Parameter(div_term[:-1], requires_grad=False)

    def forward(self, batch: PytorchBatch) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: The input batch to process.

        Returns:
            The temporal position embeddings tensor of shape (bsz, seq_len)
        """

        t = time_from_deltas(batch) if batch.get("time", None) is None else batch["time"]
        bsz, seq_len = t.shape
        device = t.device

        # First, we go from deltas to time values and unsqueeze it for broadcasting through the hidden dim.
        t = t.unsqueeze(-1)

        # temporal_embeddings will be our output container.
        # It should have shape (batch size, sequence length, embedding dim), and be on the same device as the
        # timepoints.
        temporal_embeddings = torch.zeros(bsz, seq_len, self.embedding_dim, device=device)

        temporal_embeddings[:, :, 0::2] = torch.sin(t * self.sin_div_term.unsqueeze(0).unsqueeze(0))
        temporal_embeddings[:, :, 1::2] = torch.cos(t * self.cos_div_term.unsqueeze(0).unsqueeze(0))

        return temporal_embeddings


class ConditionallyIndependentPointProcessInputLayer(torch.nn.Module):
    """Processes input batch and produces event embeddings.

    This layer accepts a batch from an event-stream PyTorch dataset and returns input embeddings from it. This
    is designed for conditionally independent models, as it does not split the input embeddings into different
    components corresponding to different dependency graph positions. Combines time and data embeddings.

    Args:
        config: Configuration parameters for the structured transformer.
    """

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__()

        self.config = config
        self.data_embedding_layer = DataEmbeddingLayer(
            n_total_embeddings=config.vocab_size,
            out_dim=config.hidden_size,
            categorical_embedding_dim=config.categorical_embedding_dim,
            numerical_embedding_dim=config.numerical_embedding_dim,
            static_embedding_mode=config.static_embedding_mode,
            split_by_measurement_indices=None,
            do_normalize_by_measurement_index=config.do_normalize_by_measurement_index,
            static_weight=config.static_embedding_weight,
            dynamic_weight=config.dynamic_embedding_weight,
            categorical_weight=config.categorical_embedding_weight,
            numerical_weight=config.numerical_embedding_weight,
        )
        self.time_embedding_layer = TemporalPositionEncoding(embedding_dim=config.hidden_size)
        self.embedding_dropout = torch.nn.Dropout(p=config.input_dropout)

    def forward(self, batch: PytorchBatch) -> torch.Tensor:
        """Returns input event embeddings for the provided batch.

        Args:
            batch: A PytorchBatch instance containing input data.
        """

        data_embed = self.data_embedding_layer(batch)
        time_embed = self.time_embedding_layer(batch)
        embed = data_embed + time_embed

        if batch.event_mask is not None:
            embed = torch.where(
                batch.event_mask.unsqueeze(-1).expand_as(embed), embed, torch.zeros_like(embed)
            )

        return self.embedding_dropout(embed)


class ConditionallyIndependentRetreivalAugTransformer(StructuredTransformerPreTrainedModel):
    """A transformer model specifically for conditionally independent point processes.

    This model uses an input layer to generate embeddings from an event-stream PyTorch dataset, and
    an InnerBlock layer for non-structured processing. As a conditionally independent model, all event
    covariates are predicted simultaneously from the history embedding.

    Args:
        config: Configuration parameters for the structured transformer.

    Raises:
        ValueError: If the provided configuration indicates a nested attention model.
    """

    def __init__(self, config: StructuredTransformerConfig):
        super().__init__(config)
        if config.chunked_cross_attention_chunk_len is None:
            raise ValueError("chunked_cross_attention_chunk_len must be provided for retreival models")

        self.config = config
        self.embed_dim = config.hidden_size
        self.input_layer = ConditionallyIndependentPointProcessInputLayer(config)
        if config.structured_event_processing_mode != StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
            raise ValueError(f"{config.structured_event_processing_mode} invalid!")

        self.non_retreival_layers = nn.ModuleList(
            [TransformerBlock(config, layer_id=i, is_seq=True, is_retreival_block=False)
             for i in range(config.retreival_layer_idx)]
        )
        self.retreival_layers = nn.ModuleList(
            [TransformerBlock(config, layer_id=i, is_seq=True, is_retreival_block=True)
             for i in range(config.retreival_layer_idx, config.num_hidden_layers)]
        )

        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.post_retreival_proj = nn.Identity()
        if config.retreived_states_hidden_size != 0:
            self.post_retreival_proj = nn.Linear(config.retreived_states_hidden_size, self.embed_dim)

        self.ln_retreival_hiddens = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def reshape_to_retreival_queries(
            self,
            hidden_states,
            *,
            allow_padding,
        ):
            """
            This function is not a part of the half_forward,
            because during generation we will mostly have sequences of length%chunk_len != 0 and it's expendive to always pad

            `allow_padding` is **intentionally left without a default value**, provide it explicitly

            Args:
                hidden_states: Tensor of shape (batch_size, seq_len, hidden)
                allow_padding: Whether to allow padding the input to be divisible by chunk_len
            Returns:
                retreival_queries: Tensor of shape (batch_size * n_chunks, hidden)
            """
            batch_size, seq_len, hidden = hidden_states.shape
            chunk_len = self.config.chunked_cross_attention_chunk_len
            n_chunks = seq_len // chunk_len
            if n_chunks * chunk_len != seq_len:
                if not allow_padding: raise RuntimeError(f"Can't reshape to retreival queries without padding, seq_len: {hidden_states.shape=}, chunk_len: {chunk_len}")
                # Pad the input to be divisible by chunk_len
                add_len = (n_chunks + 1) * chunk_len - seq_len
                hidden_states = torch.cat((hidden_states, hidden_states.new_zeros(batch_size, add_len, hidden)), dim=1)
                n_chunks += 1

            retreival_queries = hidden_states.reshape(
                batch_size, -1, self.config.chunked_cross_attention_chunk_len, hidden
            )
            retreival_queries = torch.max(retreival_queries, dim=2).values
            retreival_queries = retreival_queries.flatten(0, 1)
            return retreival_queries

    def first_half_forward(
        self,
        batch: PytorchBatch | None = None,
        *,
        input_embeds: torch.Tensor | None = None,
        past: tuple[torch.FloatTensor] | None = None,
        seq_attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> RetreivalTransformerMiddleLayersOutput:
        """Peforms partial forward of the model to get the query embeddings for retreival

        Args:
            batch: A PytorchBatch instance containing input data.
            input_embeds: Precomputed embeddings for the input data.
            past: Past hidden states for more efficient decoding.
            seq_attention_mask: Mask for the sequential attention mechanism.
            head_mask: Mask to nullify selected heads of the self-attention module.
            use_cache: Specifies whether caching should be used.
            output_attentions: Specifies whether attention probabilities should be returned in the output.
            output_hidden_states: Specifies whether hidden states should be returned in the output.

        Returns:
            RetreivalTransformerMiddleLayersOutput with the last_hidden being the hidden of the last non-retrieval layer
        """
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states

        if past is None:
            past = tuple([None] * self.config.num_hidden_layers)

        if input_embeds is None:
            assert batch is not None

            input_embeds = self.input_layer(batch)
        else:
            assert batch is None, "Can't specify both input_embeds and batch."

        torch._assert(~torch.isnan(input_embeds).any(), f"{torch.isnan(input_embeds).sum()} NaNs in input_embeds")

        if seq_attention_mask is None and batch is not None and batch.get("event_mask", None) is not None:
            seq_attention_mask = expand_mask(batch["event_mask"], input_embeds.dtype)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = input_embeds

        current_key_values = () if use_cache else None

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_kv_cache) in enumerate(zip(self.non_retreival_layers, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing:
                raise NotImplementedError()

            hidden_states, extra_return_info = layer(
                hidden_states=hidden_states,
                attention_mask=seq_attention_mask,
                layer_past=layer_kv_cache,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                retreived_hidden_states=None,  # retreived_hidden_states are not used in the first half of the model
            )

            if batch is not None and batch.event_mask is not None:
                hidden_states = torch.where(
                    batch.event_mask.unsqueeze(-1).expand_as(hidden_states),
                    hidden_states,
                    torch.zeros_like(hidden_states),
                )

            if use_cache:
                current_key_values = current_key_values + (extra_return_info["present_key_value"],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (extra_return_info["attn_weights"],)
        # end of for-loop over layers

        return RetreivalTransformerMiddleLayersOutput(
            last_hidden_state=hidden_states,
            past_key_values=current_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            all_hidden_states=all_hidden_states,
            all_self_attentions=all_self_attentions,
        )

    def second_half_forward(
        self,
        batch: PytorchBatch | None = None,
        *,
        hidden_states: torch.Tensor | None = None,
        retreived_hidden_states: torch.Tensor | None = None,
        past: tuple[torch.FloatTensor] | None = None,
        seq_attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        # below are the parameters that need to be piped from
        # the first half of the model
        first_half_output: RetreivalTransformerMiddleLayersOutput | None = None,
        all_hidden_states: tuple[torch.FloatTensor] | None = None,
        all_self_attentions: tuple[torch.FloatTensor] | None = None,
        current_key_values: tuple[torch.FloatTensor] | None = None,
    ) -> TransformerOutputWithPast:
        """Takes hidden_states and retreival_hidden_states as input and forwards the rest of the model.

        You can provide either hidden_states, all_hidden_states and all_attentions or first_half_output as input.
        Additionally takes `all_hidden_states` and `all_attentions` as input, which are the outputs of the first
        (non-retreival) layers of the model. These are only needed if `output_hidden_states` or `output_attentions`
        """
        # ############################################################
        # Input verification/formation
        if retreived_hidden_states is None:
            retreived_hidden_states = batch.retreived_hidden_states

        if retreived_hidden_states is None:
            raise ValueError("retreived_hidden_states must be provided for the second half of the model layers")

        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states

        if first_half_output is not None:
            hidden_states = first_half_output.last_hidden_state
            all_hidden_states = first_half_output.all_hidden_states
            all_self_attentions = first_half_output.all_self_attentions
            current_key_values = first_half_output.past_key_values

        if hidden_states is None:
            raise ValueError("hidden_states must be provided for the second half of the model layers")

        if output_hidden_states and all_hidden_states is None:
            raise ValueError("all_hidden_states must be provided if output_hidden_states is True")

        if past is None:
            past = tuple([None] * self.config.num_hidden_layers)

        if seq_attention_mask is None and batch is not None and batch.get("event_mask", None) is not None:
            seq_attention_mask = expand_mask(batch["event_mask"], hidden_states.dtype)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # End of input verification/formation
        # ############################################################

        retreived_hidden_states = self.post_retreival_proj(retreived_hidden_states)
        retreived_hidden_states = self.ln_retreival_hiddens(retreived_hidden_states)

        for i, (layer, layer_kv_cache) in enumerate(zip(self.retreival_layers, past[len(self.non_retreival_layers):])):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing:
                raise NotImplementedError()

            hidden_states, extra_return_info = layer(
                hidden_states=hidden_states,
                attention_mask=seq_attention_mask,
                layer_past=layer_kv_cache,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                retreived_hidden_states=retreived_hidden_states,
            )

            if batch is not None and batch.event_mask is not None:
                hidden_states = torch.where(
                    batch.event_mask.unsqueeze(-1).expand_as(hidden_states),
                    hidden_states,
                    torch.zeros_like(hidden_states),
                )

            if use_cache:
                current_key_values = current_key_values + (extra_return_info["present_key_value"],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (extra_return_info["attn_weights"],)

        hidden_states = self.ln_f(hidden_states)

        # I don't know what this is supposed to do
        # hidden_states = hidden_states.view(input_embeds.size())

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return TransformerOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=current_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def forward(
        self,
        batch: PytorchBatch | None = None,
        *,
        retreived_hidden_states: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        past: tuple[torch.FloatTensor] | None = None,
        seq_attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> TransformerOutputWithPast:
        """In this forward we assume that we have retreived hidden states already.
        If it's not the case, you need to first perform the first_half_forward,
        retreive the hidden states and then perform the second_half_forward.

        Args:
            batch: A PytorchBatch instance containing input data.
            input_embeds: Precomputed embeddings for the input data. Currently unused.
            past: Past hidden states for more efficient decoding.
            seq_attention_mask: Mask for the sequential attention mechanism.
            head_mask: Mask to nullify selected heads of the self-attention module.
            use_cache: Specifies whether caching should be used.
            output_attentions: Specifies whether attention probabilities should be returned in the output.
            output_hidden_states: Specifies whether hidden states should be returned in the output.

        Returns:
            A tuple containing hidden states, or a TransformerOutputWithPast object if return_dict is True.
        """

        non_retreival_output = self.first_half_forward(
            batch=batch,
            input_embeds=input_embeds,
            past=past,
            seq_attention_mask=seq_attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        retreival_output = self.second_half_forward(
            batch=batch,
            hidden_states=non_retreival_output.last_hidden_state,
            retreived_hidden_states=retreived_hidden_states,
            past=past,
            seq_attention_mask=seq_attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            all_hidden_states=non_retreival_output.hidden_states,
            all_self_attentions=non_retreival_output.attentions,
            current_key_values=non_retreival_output.past_key_values,
        )

        return retreival_output
