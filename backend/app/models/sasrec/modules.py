import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    # https://hongl.tistory.com/236
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        Tensorflow 스타일로 만들었다는건가?
        https://sonsnotation.blogspot.com/2020/11/8-normalization.html url 참고
        """
        super(LayerNorm, self).__init__()
        # hidden_size 크기의 1로만 구성된 tensor for weight 생성하고, parameter로 등록
        self.weight = nn.Parameter(torch.ones(hidden_size))  # [H]
        # hidden_size 크기의 0으로만 구성된 tensor for bias 생성하고, parameter로 등록
        self.bias = nn.Parameter(torch.zeros(hidden_size))  # [H]
        # 아래 forward 함수의 s에 더해줄 값
        self.variance_epsilon = eps

    def forward(self, x):
        # hidden_size 차원에서 평균 계산
        # keepdim=True : 그 아랫줄에서 x와 연산 가능하도록 차원 맞춰주기 위해
        u = x.mean(-1, keepdim=True)  # [B, L, 1]
        # hidden_size 차원에서 분산 계산
        s = (x - u).pow(2).mean(-1, keepdim=True)  # [B, L, 1]
        # Normalzie / s의 원소 값이 0인 경우를 대비하기 위해 variance_epsilon 더해줌
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)  # [B, L, H]
        # scale and shift : activate function에 적합한 분포를 갖게 하기 위함
        return self.weight * x + self.bias  # [B * L * H]


class Embeddings(nn.Module):
    """Construct the embeddings from item, position."""

    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads)
            )
        # 이 과정 왜 필요하지
        # 위에서 저렇게 ValueError 일으킬거면, 어차피 all_head_size = hidden_size 아닌가
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # generate Query, Key, Value vectors with nn.Linear
        # dimension : hidden_size -> all_head_size(hidden_size와 동일.)
        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        # Dropout (for Attention Layer)
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        # Dense Layer
        # Dimension: hidden_size -> hidden_size
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        # Layer Normalization
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        # Dropout (for Output Layer)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        """
        summary : 차원을 바꾸는 함수입니다. hidden 값을 2분할합니다.
        Args: [B, L, H]
        Returns: [B, 2, L, H // 2]
        """        
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, # defalut = 2
            self.attention_head_size, # hidden_size // num_attention_head = 32(defalut)
        ) 
        x = x.view(*new_x_shape)  # [B, L, (num_att_head), (att_head_size)]
        return x.permute(0, 2, 1, 3) # [B, L, 2, H // 2] => [B, 2, L, H // 2]  ( [B, (num_att_head), L, (att_head_size)] )

    def forward(self, input_tensor, attention_mask):
        """_summary_
        Args:
            input_tensor (tenser): (batch * max_len * hidden_size)
            attention_mask (tenser): (batch * 1 * 1 * max_len) or (batch * 1 * max_len * max_len)

        Returns:
            hidden_states (tensor): (batch * max_len * hidden_size)
        """        
        mixed_query_layer = self.query(input_tensor)  # [B, L, H]
        mixed_key_layer = self.key(input_tensor)  # [B, L, H]
        mixed_value_layer = self.value(input_tensor)  # [B, L, H]

        # [B, L, H] => [B, 2, L, H // 2], hidden 값을 2분할합니다. = [B * (num_att_head) * L * (att_head_size)]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # [B * (num_att_head) * L * (att_head_size)] * [B * (num_att_head) * (att_head_size) * L]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [B * (num_att_head) * L * L]

        # attention_head_size = hidden // 2, [B, 2, L, H // 2] => [B, 2, L, L]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [B * (num_att_head) * L * L]
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size, heads, seq_len, seq_len] scores
        # [batch_size, 1, 1, seq_len], 패딩 값은 마이너스 거의 무한대. (train에선 [batch_size, 1, seq_len, seq_len])
        attention_scores = attention_scores + attention_mask  # [B * (num_att_head) * L * L]

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [B * (num_att_head) * L * L]
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)  # [B * (num_att_head) * L * L]
        # [B, 2, L, L] * [B, 2, L, H // 2] => [B, 2, L, H // 2]
        context_layer = torch.matmul(attention_probs, value_layer)  # [B * (num_att_head) * L * (att_head_size)]
        # [B, 2, L, H // 2] => [B, L, 2, H // 2]
        # contiguous => https://jimmy-ai.tistory.com/122
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [B * L * (num_att_head) * (att_head_size)]
        # [B, L, H] shape 저장.
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [B, L, 2, H // 2] => [B, L, H]
        context_layer = context_layer.view(*new_context_layer_shape)
        # dense : Linear H => H
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        # Add + Norm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states  # [B * L * H]


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str): # hidden_act : gelu(defalut)
            self.intermediate_act_fn = ACT2FN[args.hidden_act] 
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        """_summary_
        Args:
            shape of input_tensor = [B, L, H]

        Returns:
            _type_: _description_
        """
        hidden_states = self.dense_1(input_tensor)  # [B, L, H*4]
        # activate function 적용
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)  # [B, L, H]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states  # [B, L, H]


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)  # [B, L, H]
        intermediate_output = self.intermediate(attention_output)  # [B, L, H]
        return intermediate_output  # [B, L, H]


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(args.num_hidden_layers)] # num_hidden_layers : 2(default)
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)  # [B, L, H]
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
