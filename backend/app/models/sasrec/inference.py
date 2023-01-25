import argparse
import os

import pandas as pd

import torch
from torch.utils.data import DataLoader, SequentialSampler

from datasets import SASRecDataset
from models import S3RecModel
from utils import set_seed
from trainers import test_score
 

def main():
    parser = argparse.ArgumentParser()

    # 데이터 경로와 네이밍 부분.
    parser.add_argument("--data_dir", default="../data/", type=str)
    # parser.add_argument("--output_dir", default="output/", type=str)
    # parser.add_argument("--data_name", default="Ml", type=str)
    # parser.add_argument("--do_eval", action="store_true")

    # # 모델 argument(하이퍼 파라미터)
    # parser.add_argument("--model_name", default="Finetune_full", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="hidden size of transformer model"
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
    parser.add_argument("--max_seq_length", default=100, type=int)

    # # train args, 트레이너 하이퍼파라미터
    parser.add_argument(
        "--batch_size", type=int, default=1, help="number of batch_size"
    )
    parser.add_argument("--no_cuda", action="store_true")
    # parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")


    # parser 형태로 argument를 입력받습니다.
    args = parser.parse_args()

    # 시드 고정 (utils.py 내 함수 존재)
    set_seed(args.seed)

    # GPU 관련 설정 해줍니다.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device("cuda" if args.cuda_condition else "cpu")

    # 데이터 파일 불러오는 경로 설정합니다.
    args.data_file = args.data_dir + "user.csv"
    user_seqs = pd.read_csv(args.data_file)
    rest_info = pd.read_csv(args.data_dir + "rest.csv")

    user_seq = user_seqs['rest_code'][0]  # [2062 2840  875 2841 2867 2855 2846    1 2839 2460 1841 2845 2872 1013]
    user_seq = user_seq[1:-1].split()
    user_seq = [int(num) for num in user_seq]
    user_id = user_seqs['user_code'][0]  # 0

    max_item = int(max(rest_info['rest_code']))
    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    '''
    user_id : 해당 유저 encoding 값
    user_seq : 해당 유저 음식점 리뷰 시퀀셜 기록
    args.item_size : item encoding 최대값 + 2
    '''
    # args.mask_id = max_item + 1

    submission_dataset = SASRecDataset(args, user_seq)
    submission_sampler = SequentialSampler(submission_dataset)
    submission_dataloader = DataLoader(
        submission_dataset, sampler=submission_sampler, batch_size=args.batch_size
    )

    model = S3RecModel(args=args)
    model = model.to(args.device)
    # 트레이너에 load 함수 사용해 모델 불러옵니다.
    model.load_state_dict(torch.load('../data/SASRec-0124.pt'))

    pred = test_score(args, 1, submission_dataloader, model)  # epoch 1로 넣음
    

if __name__ == "__main__":
    main()
