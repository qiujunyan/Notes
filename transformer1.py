import copy
import math
from abc import ABC

import torch
from torch import nn


def clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransConfig(object):
  def __init__(self, **config):
    '''config = TransConfig(init_word_embedding=False,
                            vocab_size=None,
                            hidden_size=None,
                            num_layers=None,
                            num_heads=None,
                            intermediate_size=None,
                            max_position_size=512,
                            token_type_size=2,
                            layer_norm_eps=1e-10,
                            dropout=0.0,
                            is_decoder=False,
                            casual_mask=False,
                            neg_inf=-1e10,
                            eps=1e-10)'''
    self.init_word_embedding = config.get("init_word_embedding", False)
    self.vocab_size = config.get("vocab_size", None)
    self.hidden_size = config.get("hidden_size", None)
    self.num_layers = config.get("num_layers", None)
    self.num_heads = config.get("num_heads", None)
    self.intermediate_size = config.get("intermediate_size", None)
    self.max_position_size = config.get("max_position_size", 512)
    self.token_type_size = config.get("token_type_size", 2)
    self.layer_norm_eps = config.get("layer_norm_eps", 1e-10)
    self.dropout = config.get("dropout", 0.0)
    self.is_decoder = config.get("is_decoder", False)
    self.casual_mask = config.get("casual_mask", self.is_decoder)
    self.init_range = config.get("init_range", 0.02)
    self.neg_inf=config.get("neg_inf", -1e10)
    self.eps=config.get("eps", 1e-10)


class TransModel(nn.Module):
  def __init__(self, config: TransConfig):
    super(TransModel, self).__init__()
    self.encoder = TransEncoder(config)
    self.num_layers = config.num_layers
    self._is_decoder = config.is_decoder
    self._casual_mask = config.casual_mask
    self.embeddings = TransEmbedding(config)
    self.pooler = TransPooler(config)
    self.init_range = config.init_range

    self.init_weights()

  def forward(self, input_ids: torch.LongTensor = None,
              encoder_states: torch.FloatTensor = None,
              attention_mask=None,
              encoder_attention_mask=None,
              token_type_ids: torch.LongTensor = None,
              position_ids: torch.LongTensor = None,
              inputs_embeds: torch.FloatTensor = None):
    """
    :param input_ids: ids of target sequence
    :param encoder_states: outputs of transformer encoder, useful when model is explained as a decoder
    :param attention_mask: self attention mask, When bool tensor is provided and a value is True, the corresponding value on
    the attention layer will be ignored. Otherwise, the non-zero corresponded value will be ignored.
    :param encoder_attention_mask: cross attention mask, When a value is True, the corresponding value on
    the attention layer will be ignored. Otherwise, the non-zero corresponded value will be ignored.
    :param token_type_ids: [optional]
    :param position_ids: [optional]
    :param inputs_embeds: embeddings of source sequence, useful when config.init_word_embedding is False
    :return:

    shape:
    -Inputs:
    -input_ids: batch_size * tgt_seq_lens
    -encoder_states: batch_size * src_seq_lens * hidden_size
    -attention_mask: batch_size * tgt_seq_lens
    -encoder_attention_mask: batch_size * src_seq_lens * tgt_seq_lens
    -token_type_ids:
    -position_ids:
    inputs_embeds: batch_size * tgt_seq_lens * hidden_size

    -Outputs:
    -encoder_outputs: batch_size * tgt_seq_lens*  hidden_size
    -pooler_outputs: batch_size * hidden_size
    """

    if input_ids is not None:
      input_shape = input_ids.size()
    elif inputs_embeds is not None:
      input_shape = inputs_embeds.size()[:-1]
    else:
      raise ValueError("Either input_ids or inputs_embeds must be specified")

    extended_attention_mask = self.get_extended_mask(attention_mask, input_shape)
    if encoder_attention_mask is not None:
      extended_encoder_attention_mask = self.get_inverted_mask(encoder_attention_mask)
    else:
      extended_encoder_attention_mask = None

    embedding_output = self.embeddings(input_ids=input_ids,
                                       input_embeds=inputs_embeds,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids)
    encoder_outputs = self.encoder(hidden_states=embedding_output,
                                   attention_mask=extended_attention_mask,
                                   encoder_hidden_states=encoder_states,
                                   encoder_attention_mask=extended_encoder_attention_mask)
    pooler_ourput = self.pooler(encoder_outputs)
    return encoder_outputs, pooler_ourput

  def get_extended_mask(self, attention_mask, input_shape):
    # attention_mask: FloatTensor (batch_size, to_seq_length) -> (batch_size, 1, from_seq_length, to_seq_length)
    # input_shape: tuple
    device = attention_mask.device
    if self._is_decoder and self._casual_mask:
      batch_size, seq_length = input_shape
      causal_mask = torch.ones([batch_size, seq_length, seq_length], device=device)
      causal_mask = torch.tril(causal_mask, diagonal=0)
      extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    else:
      extended_attention_mask = attention_mask[:, None, None, :]

    if isinstance(extended_attention_mask.cpu(), torch.BoolTensor):
      extended_attention_mask = ~extended_attention_mask
    else:
      extended_attention_mask = 1 - extended_attention_mask
      extended_attention_mask = extended_attention_mask.bool()
    return extended_attention_mask

  @staticmethod
  def get_inverted_mask(encoder_attention_mask):
    extended_encoder_attention_mask = encoder_attention_mask[:, None, None, :]
    extended_encoder_attention_mask = 1 - extended_encoder_attention_mask
    extended_encoder_attention_mask = extended_encoder_attention_mask.bool()
    return extended_encoder_attention_mask

  def init_weights(self):
    """ Initialize the weights """
    for module in self.modules():
      if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=self.init_range)
      elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
      if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class TransEncoder(nn.Module, ABC):
  def __init__(self, config: TransConfig):
    super(TransEncoder, self).__init__()
    self.layer = clones(TransLayer(config), config.num_layers)

  def forward(self, hidden_states,
              attention_mask: torch.BoolTensor,
              encoder_hidden_states,
              encoder_attention_mask: torch.BoolTensor):
    r"""
    attention_mask: (batch_size, seq_lens), if a value is True,
    the correspoding value on the attention layer will be ignored.
    """
    for i, layer in enumerate(self.layer):
      hidden_states = layer(hidden_states,
                            attention_mask,
                            encoder_hidden_states,
                            encoder_attention_mask)
    return hidden_states


class TransEmbedding(nn.Module):
  def __init__(self, config: TransConfig):
    super(TransEmbedding, self).__init__()
    self.init_word_embedding = config.init_word_embedding
    if self.init_word_embedding:
      self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
    self.position_embeddings = nn.Embedding(config.max_position_size, config.hidden_size)
    self.token_type_embeddings = nn.Embedding(config.token_type_size, config.hidden_size)
    self.register_buffer("position_ids", torch.arange(config.max_position_size).expand((1, -1)))

    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, input_ids=None, token_type_ids=None, position_ids=None, input_embeds=None):
    if input_ids is not None:
      input_shape = input_ids.size()
    else:
      input_shape = input_embeds.size()[:-1]
    seq_length = input_shape[1]

    if position_ids is None:
      position_ids = self.position_ids[:, :seq_length]

    if token_type_ids is None:
      token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)

    if input_embeds is None and self.init_word_embedding:
      input_embeds = self.word_embeddings(input_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    position_embeddings = self.position_embeddings(position_ids)
    embeddings = input_embeds + token_type_embeddings + position_embeddings

    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


class TransLayer(nn.Module):
  def __init__(self, config):
    super(TransLayer, self).__init__()
    self.attention = TransAttention(config)
    self._is_decoder = config.is_decoder
    if self._is_decoder:
      self.cross_attention = TransAttention(config)
    self.intermediate = TransIntermediate(config)
    self.output = TransOut(config)

  def forward(self, hidden_states,
              attention_mask=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None):
    attention_outputs = self.attention(hidden_states, attention_mask)
    if self._is_decoder:
      attention_outputs = self.cross_attention(attention_outputs,
                                               attention_mask,
                                               encoder_hidden_states,
                                               encoder_attention_mask)
    intermediate_output = self.intermediate(attention_outputs)
    layer_output = self.output(intermediate_output, attention_outputs)
    return layer_output


class TransIntermediate(nn.Module):
  def __init__(self, config: TransConfig):
    super(TransIntermediate, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.activation_fn = nn.GELU()

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.activation_fn(hidden_states)
    return hidden_states


class TransOut(nn.Module):
  def __init__(self, config: TransConfig):
    super(TransOut, self).__init__()
    self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states


class TransSelfOut(nn.Module):
  def __init__(self, config: TransConfig):
    super(TransSelfOut, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states


class TransAttention(nn.Module):
  def __init__(self, config: TransConfig):
    super(TransAttention, self).__init__()
    self.self = TransSelfAtention(config)
    self.output = TransSelfOut(config)

  def forward(self, hidden_states,
              attention_mask=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None):
    self_outputs = self.self(hidden_states,
                             attention_mask,
                             encoder_hidden_states,
                             encoder_attention_mask)
    attention_output = self.output(self_outputs, hidden_states)
    return attention_output


class TransSelfAtention(nn.Module):
  def __init__(self, config: TransConfig):
    super(TransSelfAtention, self).__init__()
    assert config.hidden_size % config.num_heads == 0
    self.neg_inf = config.neg_inf
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_heads
    self.head_size = config.hidden_size // config.num_heads

    self.query = nn.Linear(config.hidden_size, config.hidden_size)
    self.key = nn.Linear(config.hidden_size, config.hidden_size)
    self.value = nn.Linear(config.hidden_size, config.hidden_size)

    self.dropout = nn.Dropout(config.dropout)

  def forward(self, hidden_states,
              attention_mask=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None):
    query_states = self.query(hidden_states)

    if encoder_hidden_states is not None:
      key_states = self.key(encoder_hidden_states)
      value_states = self.value(encoder_hidden_states)
      attention_mask = encoder_attention_mask
    else:
      key_states = self.key(hidden_states)
      value_states = self.key(hidden_states)

    query_states = self.split_and_transpose(query_states)
    key_states = self.split_and_transpose(key_states)
    value_states = self.split_and_transpose(value_states)

    attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.head_size)

    if attention_mask is not None:
      attention_scores = attention_scores.masked_fill(attention_mask, self.neg_inf)
    attention_probs = torch.softmax(attention_scores, -1)
    attention_probs = self.dropout(attention_probs)

    context_states = torch.matmul(attention_probs, value_states)
    context_states = context_states.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_states.size()[:-2] + (self.hidden_size,)
    attention_output = context_states.view(*new_context_layer_shape)

    return attention_output

  def split_and_transpose(self, hidden_states):
    new_x_shape = hidden_states.size()[:-1] + (self.num_heads, self.head_size)
    hidden_states = hidden_states.view(*new_x_shape)
    return hidden_states.permute(0, 2, 1, 3)


class TransPooler(nn.Module):
  def __init__(self, config: TransConfig):
    super(TransPooler, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.activation = nn.Tanh()

  def forward(self, hidden_states):
    first_token_tensor = hidden_states[:, 0]
    pooled_output = self.dense(first_token_tensor)
    pooled_output = self.activation(pooled_output)
    return pooled_output


class TransScheduler(object):
  def __init__(self, optimizer, warmup, lr: list = None, d_model=None):
    self.optimizer = optimizer
    self.warmup = warmup
    self.lr0 = lr
    self.d_model = d_model
    self.num_groups = len(optimizer.param_groups)

    assert lr or d_model
    if lr is not None:
      assert len(lr) == self.num_groups

  def __call__(self, step_num):
    step_num += 1
    if self.lr0 is None:
      lrs = [self.d_model ** -0.5] * self.num_groups
    else:
      lrs = self.lr0

    for i, lr in enumerate(lrs):
      lr_rate = lr * min(step_num ** -0.5, step_num * self.warmup ** -1.5)
      self.optimizer.param_groups[i]['lr'] = lr_rate
