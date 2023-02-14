import torch
import torch.nn as nn

from .modules import Encoder, LayerNorm


class S3RecModel(nn.Module):
    def __init__(self, args):
        super(S3RecModel, self).__init__()
        # item embedding
        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0 # hidden_size : 64(defalut)
        )
        
        # positional embedding
        # label 개수 <= max_seq_length
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        # 트랜스포머의 일부 구성요소 시용한 인코더(사실은 디코더와 유사하다고 함.)
        # modules에 빡세게 구현되어있음.
        self.item_encoder = Encoder(args)
        # layer normalization
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        # dropout
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.hidden_size = args.hidden_size

        # add unique dense layer for 4 losses respectively
        self.criterion = nn.BCELoss(reduction="none")
        # initialize layers
        self.apply(self.init_weights)


    def add_position_embedding(self, sequence):
        """_summary_
        입력 값에서 아이템 임베딩을 해준 뒤 포지션 임베딩을 더해줍니다.
        Args:
            sequence (tenser): (batch * max_len), 영화 id 기록
        Returns:
            sequence_emb (tenser): (batch * max_len * hidden_size)
        """        
        seq_length = sequence.size(1)  # max_len
        '''
        tensor([ 0, 1, ..., seq_length-1 ], device='cuda:0')
        '''
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        '''
        [B * L]
        tensor([[ 0, 1, ..., seq_length-1 ],
        [...] * (batch_szie-2),
        [ 0, 1, ..., seq_length-1 ]], device='cuda:0')
        '''
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        # 아이템 임베딩 먼저하고 (B * L => B * L * H)
        item_embeddings = self.item_embeddings(sequence)
        # 포지션 임베딩 진행. (B * L => B * L * H)
        position_embeddings = self.position_embeddings(position_ids)
        # 아이템 임베딩 + 포지션 임베딩
        sequence_emb = item_embeddings + position_embeddings
        # Layer Nomralization [B * L * H]
        sequence_emb = self.LayerNorm(sequence_emb)
        # Dropout (결과 tensor 값들 일부 0으로 바뀜)
        sequence_emb = self.dropout(sequence_emb)  # [B * L * H]

        return sequence_emb


    # Fine tune
    # same as SASRec
    def finetune(self, input_ids):
        # attention_mask : [B, L], 패딩된 값은 0 / 아닌 값은 1인 마스킹 행렬 만들기.
        attention_mask = (input_ids > 0).long()
        # extended_attention_mask : [B, 1, 1, L]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        # subsequent_mask : 상 삼각행렬. [[[0, 1, 1, .. 1], [0, 0, 1, .. 1], ... , [0,0, ... 0]]], [1, L, L]
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        # subsequent_mask : 하 삼각행렬. [[[[1, 0, 0, .. 0], [1, 1, 0, .. 0], ... , [1,1, ... 1]]]], [1, 1, L, L]
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        # [B, 1, 1, L] * [1, 1, L, L] => [B, 1, L, L]
        # extended_attention_mask : 패딩아닌 것만 1
        # subsequent_mask : 하나의 시퀀셜 영화기록 L을 L * L로 확장. 이전 기록 마스킹 하는 식으로 확장.
        # 두 마스킹을 곱하면 패딩과 이전 기록 마스킹을 동시에 하는 마스킹 텐서 탄생.
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        # 마스킹 된 값 -10000 곱하기. 마스킹 안된 값은 0.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # Linear & Embedding weight mean=0, std=initializer_range로 초기화
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            # LayerNorm weight=1, bias=0 으로 초기화
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # Bias 존재하는 Linear bias=0 으로 초기화
            module.bias.data.zero_()
