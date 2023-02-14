import argparse
import os

import torch

from .models import S3RecModel
from .utils import set_seed
from .trainers import trainers


def recommend(user_seq: list, item_candidate : list, max_item = 41460):
    """
    Args:
        user_seq (str(list)): 해당 유저가 방문한 rest_code 리스트. str 로 묶여서 옴.
        item_candidate (list): 아이템 후보(x,y 또는 속성으로 걸러진) rest_code 리스트.
        max_item (int, optional): 음식점 개수, Defaults to 41460.

    Returns:
        pred (list) : Top 3 rest_code list
    """    
    parser = argparse.ArgumentParser()

    # 데이터 경로와 네이밍 부분.
    parser.add_argument("--data_dir", default="app/models/data/", type=str) # ./models/data/
    parser.add_argument("--model_name", default="SASRec-time", type=str)

    # # 모델 argument(하이퍼 파라미터)
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    # # 활성화 함수. (default gelu => relu 변형)
    parser.add_argument("--hidden_act", default="gelu", type=str)

    # dropout하는 prob 기준 값 정하기? (모델 본 사람이 채워줘.)
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.2,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p"
    )

    # # 모델 파라미터 initializer 범위 설정? (모델 본 사람이 채워줘.)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    # # 최대 시퀀셜 길이 설정
    parser.add_argument("--max_seq_length", default=150, type=int)

    # # train args, 트레이너 하이퍼파라미터
    parser.add_argument(
        "--batch_size", type=int, default=1, help="number of batch_size"
    )
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--k", type=int, default=20, help="top k recommend")


    # parser 형태로 argument를 입력받습니다.
    args = parser.parse_args()

    # 시드 고정 (utils.py 내 함수 존재)
    set_seed(args.seed)

    # GPU 관련 설정 해줍니다.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device("cuda" if args.cuda_condition else "cpu")

    # user_seq = user_seqs['rest_code'][0]  # [2062 2840  875 2841 2867 2855 2846    1 2839 2460 1841 2845 2872 1013]
    user_seq = eval(user_seq)  # str -> list

    ################# item max 값 받아오는 부분 나중에 처리 필요 (일단 임시로 csv 파일로 처리)
    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    ################################################################

    model = S3RecModel(args=args)
    model = model.to(args.device)
    # 트레이너에 load 함수 사용해 모델 불러옵니다.
    model.load_state_dict(torch.load(args.data_dir + args.model_name + '.pt', map_location=torch.device(args.device)))

    pred = trainers(args, user_seq, model, item_candidate)
    return pred
