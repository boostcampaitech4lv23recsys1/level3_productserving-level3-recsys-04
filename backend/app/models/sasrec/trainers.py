import numpy as np
import torch
from tqdm import tqdm


def test_score(args, epoch, dataloader, model):
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
        user_ids, input_ids, _, target_neg, answers = batch
        recommend_output = model.finetune(input_ids)  # [B, L, H]

        recommend_output = recommend_output[:, -1, :]  # [B, H]

        rating_pred = predict_full(recommend_output, model)  # [B, item_num]

        rating_pred = rating_pred.cpu().data.numpy().copy()
        batch_user_index = user_ids.cpu().numpy()

        # 해당 User가 이미 방문한 음식점 제외(마스킹)
        rating_pred[args.train_matrix[batch_user_index].toarray() > 0] = -np.inf

        # mask_id 제외(마스킹)
        rating_pred[:, -1] = -np.inf

        # TOP 3 index 추출
        ind = np.argpartition(rating_pred, -3)[:, -3:]

        return ind


def trainers(args, _input, model):
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

    model.eval()

    pad_len = args.max_seq_length - len(_input)
    _input_model = [0] * pad_len + _input
    _input_model = _input_model[-args.max_seq_length:]
    _input_model = torch.tensor(_input_model, dtype=torch.long)
    _input_model = _input_model.to(args.device)
    
    recommend_output = model.finetune(_input_model.unsqueeze(0))
    recommend_output = recommend_output[:, -1, :]
    rating_pred = predict_full(recommend_output, model)
    rating_pred = rating_pred.squeeze()

    rating_pred = rating_pred.cpu().data.numpy()

    # 해당 User가 이미 방문한 음식점 제외(마스킹)
    rating_pred[_input] = -np.inf

    # mask_id 제외(마스킹)
    rating_pred[-1] = -np.inf

    # TOP 3 index 추출
    ind = np.argpartition(rating_pred, -3)[-3:]

    return ind

    
