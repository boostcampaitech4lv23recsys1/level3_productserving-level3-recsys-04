import numpy as np
import torch


def trainers(args, _input, model, item_candidate):
    """_summary_
    Args:
        args (_type_): 설정들
        _input (_type_): 사용자가 방문한 음식점 기록 리스트(rest_code 형태).
        model (Pytorch model): 모델
        item_candidate (list): x,y / tag 등으로 걸러진 후보 음식점 리스트.
    Returns:
        ind (list): 추천 음식점 리스트(rest_code 형태)
    """    
    def predict_full(seq_out, model, item_candidate):
        """_summary_
        Args:
            seq_out ([1, hidden_size]): 추천 결과물
            model : model 파일
            item_candidate(list) : 후보군 리스트 
        Returns:
            rating_pred ([1, item_candidate_num]): 유저들(batch)에 대한 모든 아이템 score
        """        
        # [item_candidate_num, hidden_size]
        test_item_emb = model.item_embeddings.weight[item_candidate,:]
        # [1, hidden_size] * [hidden_size, item_candidate_num] = [1, item_candidate_num]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred # [1, item_candidate_num]

    model.eval()

    # _input_model : 모델에 넣기 위해 _input(유저의 음식점 기록 리스트) 가공
    pad_len = args.max_seq_length - len(_input)
    _input_model = [0] * pad_len + _input
    _input_model = _input_model[-args.max_seq_length:]
    _input_model = torch.tensor(_input_model, dtype=torch.long)
    _input_model = _input_model.to(args.device)
    
    recommend_output = model.finetune(_input_model.unsqueeze(0))
    recommend_output = recommend_output[:, -1, :]

    # 이미 본 기록은 후보군에서 제외
    item_candidate = list(set(item_candidate) - set(_input))

    rating_pred = predict_full(recommend_output, model, item_candidate)
    rating_pred = rating_pred.squeeze() # [1, item_candidate_num] => [item_candidate_num]

    rating_pred = rating_pred.cpu().data.numpy()

    # TOP k index 추출
    ind = np.argpartition(rating_pred, -args.k)[-args.k:]
    item_candidate = np.array(item_candidate)
    return item_candidate[ind] # index로 rest_code로 변환해주기.
