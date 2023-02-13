import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam

from .utils import recallk


def cross_entropy(seq_out, pos_ids, neg_ids, model):
    # [batch seq_len hidden_size]
    pos_emb = model.item_embeddings(pos_ids)
    neg_emb = model.item_embeddings(neg_ids)
    # [batch*seq_len hidden_size]
    pos = pos_emb.view(-1, pos_emb.size(2))
    neg = neg_emb.view(-1, neg_emb.size(2))
    seq_emb = seq_out.view(-1, model.hidden_size)  # [batch*seq_len hidden_size]
    pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
    neg_logits = torch.sum(neg * seq_emb, -1)
    istarget = (
        (pos_ids > 0).view(pos_ids.size(0) * model.args.max_seq_length).float()
    )  # [batch*seq_len]
    loss = torch.sum(
        -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
        - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
    ) / torch.sum(istarget)

    return loss


def iteration(args, epoch, dataloader, model):

    betas = (args.adam_beta1, args.adam_beta2)
    optim = Adam(
        model.parameters(),
        lr=args.lr,
        betas=betas,
        weight_decay=args.weight_decay,
    )

    # tqdm을 통해 iter를 만듭니다.
    # 핵심은 enumerate(dataloader) 이 것만 기억하셔도 문제 없습니다.
    # 나머지 코드는 tqdm 출력을 이쁘게 도와주는 도구입니다.
    rec_data_iter = tqdm(
        enumerate(dataloader),
        desc="Recommendation EP:%d" % (epoch),
        total=len(dataloader),
        bar_format="{l_bar}{r_bar}",
    )

    # 클래스 내 모델 train 모드(파라미터 업데이트 가능)
    model.train()
    # loss 기록을 위해 만듭니다.
    rec_avg_loss = 0.0
    rec_cur_loss = 0.0

    for i, batch in rec_data_iter: # enumerate(dataloader) <=> rec_data_iter
        # 0. batch_data will be sent into the device(GPU or CPU) 
        # args.gpu_id args.cuda_condition
        batch = tuple(t.to(args.device) for t in batch)
        _, input_ids, target_pos, target_neg = batch
        # Binary cross_entropy
        sequence_output = model.finetune(input_ids)
        loss = cross_entropy(sequence_output, target_pos, target_neg, model)
        optim.zero_grad()
        loss.backward()
        optim.step()

        rec_avg_loss += loss.item()
        rec_cur_loss = loss.item()

    post_fix = {
        "epoch": epoch,
        "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_data_iter)),
        "rec_cur_loss": "{:.4f}".format(rec_cur_loss),
    }

    if (epoch + 1) % args.log_freq == 0:
        print(str(post_fix))


def test_score(args, epoch, dataloader, model, test_lines):
    # tqdm을 통해 iter를 만듭니다.
    # 핵심은 enumerate(dataloader) 이 것만 기억하셔도 문제 없습니다.
    # 나머지 코드는 tqdm 출력을 이쁘게 도와주는 도구입니다.
    rec_data_iter = tqdm(
        enumerate(dataloader),
        desc="Recommendation EP:%d" % (epoch),
        total=len(dataloader),
        bar_format="{l_bar}{r_bar}",
    )

    model.eval()

    def predict_full(seq_out, model):
        """_summary_
        Args:
            seq_out ([batch, hidden_size]): 추천 결과물
            model : model 파일
        Returns:
            rating_pred ([batch, item_num(2872)]): 유저들(batch)에 대한 모든 아이템 score
        """        
        # [item_num, hidden_size]
        test_item_emb = model.item_embeddings.weight
        # [batch, hidden_size]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred  # [B item_num]

    for i, batch in rec_data_iter:

        batch = tuple(t.to(args.device) for t in batch)
        user_ids, input_ids, _, target_neg = batch
        recommend_output = model.finetune(input_ids)  # [B, L, H]

        recommend_output = recommend_output[:, -1, :]  # [B, H]

        rating_pred = predict_full(recommend_output, model)  # [B, item_num]

        rating_pred = rating_pred.cpu().data.numpy().copy()
        batch_user_index = user_ids.cpu().numpy()

        # 해당 User가 이미 방문한 음식점 제외(마스킹)
        rating_pred[args.train_matrix[batch_user_index].toarray() > 0] = -np.inf

        # mask_id 제외(마스킹)
        rating_pred[:, -1] = -np.inf

        # TOP 20 index 추출
        ind = np.argpartition(rating_pred, -20)[:, -20:]
        # pred_list : 예측 음식점 3개, (user 개수 * 3) 
        # answer_list : 정답(test) 음식점들, (user 개수 * n(유저의 test 개수마다 바뀜))
        if i == 0:
            pred_list = ind
        else:
            pred_list = np.append(pred_list, ind, axis=0)


    # _recall : 유저마다 recall 개산하는 리스트.
    _recall = []

    for answer, pred in zip(test_lines, pred_list):
        _recall.append(recallk(answer,pred, 20))

    # 모든 유저의 recall 평균 값 반환
    return sum(_recall) / len(_recall), pred_list
