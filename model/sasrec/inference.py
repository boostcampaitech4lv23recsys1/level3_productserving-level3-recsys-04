import argparse
import os

import torch
from torch.utils.data import DataLoader, SequentialSampler

from encoding import encoding, decoding
from datasets import SASRecDataset
from models import S3RecModel
from trainers import FinetuneTrainer
from utils import (
    check_path,
    generate_submission_file,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)
 

def main():
    parser = argparse.ArgumentParser()

    # 데이터 경로와 네이밍 부분.
    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)
    parser.add_argument("--do_eval", action="store_true")

    # 모델 argument(하이퍼 파라미터)
    parser.add_argument("--model_name", default="Finetune_full", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=300, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    # 활성화 함수. (default gelu => relu 변형)
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

    # 모델 파라미터 initializer 범위 설정? (모델 본 사람이 채워줘.)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    # 최대 시퀀셜 길이 설정
    parser.add_argument("--max_seq_length", default=250, type=int)

    # train args, 트레이너 하이퍼파라미터
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # 옵티마이저 관련 하이퍼파라미터
    parser.add_argument(
        "--weight_decay", type=float, default=1e-06, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.7, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.9999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")


    # parser 형태로 argument를 입력받습니다.
    args = parser.parse_args()

    # 시드 고정 (utils.py 내 함수 존재)
    set_seed(args.seed)
    # output 폴더가 존재하는지 체크. 존재하지 않는다면 만들어줍니다. (utils.py 내 함수 존재)
    check_path(args.output_dir)

    # GPU 관련 설정 해줍니다.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # 데이터 파일 불러오는 경로 설정합니다.
    args.data_file = args.data_dir + "train_new.csv" # "train_ratings.csv"

    # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열, => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
    # max_item : 가장 큰 item_id, matrix 3개 : 유저-아이템 희소행렬
    # submission_rating_matrix : 유저-아이템 희소행렬, 유저마다 영화 시청기록은 빼지 않음.
    #user_seq, max_item, _, submission_rating_matrix = get_user_seqs(args.data_file)
    user_seq, max_item, train_matrix = get_user_seqs(args.data_file)
    # item2attribute : dict(item_id : genre의 list), attribute_size : genre id의 가장 큰 값

 

    # item, genre id의 가장 큰 값 저장합니다.
    args.item_size = max_item + 2
    args.mask_id = max_item + 1

    # save model args, (model_name : Finetune_full, data_name : Ml)
    args_str = f"{args.model_name}-{args.data_name}"

    print(str(args))


    # args.train_matrix = submission_rating_matrix

    # args_str : Finetune_full-Ml, args.output_dir : output
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    submission_dataset = SASRecDataset(args, user_seq, data_type="submission")
    submission_sampler = SequentialSampler(submission_dataset)
    submission_dataloader = DataLoader(
        submission_dataset, sampler=submission_sampler, batch_size=args.batch_size
    )

    model = S3RecModel(args=args)

    trainer = FinetuneTrainer(model, None, None, submission_dataloader, args)

    # 트레이너에 load 함수 사용해 모델 불러옵니다.
    # model.load_state_dict(torch.load(args.checkpoint_path))
    trainer.load(args.checkpoint_path)
    print(f"Load model from {args.checkpoint_path} for submission!")
    # 모델에서 예측값 받아옵니다. 0은 0 epoch 의미합니다.
    # trainers.py 내 FinetuneTrainer 클래스 iteration(0, submission_dataloader, mode="submission")함수.
    preds = trainer.submission(0)

    # utils.py 내 함수 사용. preds 값 데이터 프레임으로 변환시켜줍니다.
    generate_submission_file(args.data_file, preds)


if __name__ == "__main__":
    main()
