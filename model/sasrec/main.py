import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import SASRecTrainDataset
from models import S3RecModel
from utils import (
    check_path,
    get_user_seqs,
    set_seed,
    personalizeion
)
from trainers import (
    iteration,
    test_score
)


# 0.01687, 25에폭(time)
def main():
    parser = argparse.ArgumentParser()
    
    # 데이터 경로와 네이밍 부분.
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="0130", type=str)
    parser.add_argument("--data_type", default="rand", type=str)
    parser.add_argument("--model_name", default="SASRec", type=str)
    

    # 모델 argument(하이퍼 파라미터)
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    # 활성화 함수. (default gelu => relu 변형)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu

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
    # 모델 파라미터 initializer 범위 설정
    parser.add_argument("--initializer_range", type=float, default=0.02)
    # 최대 시퀀셜 길이 설정
    parser.add_argument("--max_seq_length", default=150, type=int)

    # train args, 트레이너 하이퍼파라미터
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs") # 200
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # 옵티마이저 관련 하이퍼파라미터
    parser.add_argument(
        "--weight_decay", type=float, default=1e-6, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.7, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.9999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # 인코딩을 사용할 것인지 체크.
    parser.add_argument("--using_encoding", action="store_true")

    # parser 형태로 argument를 입력받습니다.
    args = parser.parse_args()
    
    # 시드 고정 (utils.py 내 함수 존재)
    set_seed(args.seed)
    # output 폴더가 존재하는지 체크. 존재하지 않는다면 만들어줍니다. (utils.py 내 함수 존재)
    check_path(args.output_dir)

    
    # GPU 관련 설정 해줍니다.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device("cuda" if args.cuda_condition else "cpu")

    # 데이터 파일 불러오는 경로 설정합니다.
    path = '../data/'

    train = pd.read_csv(path + f'train_{args.data_type}.csv')    
    test = pd.read_csv(path + f'test_{args.data_type}.csv')
   
    
    # 자세한건 get_user_seqs 함수(utils.py) 내에 써놨습니다.
    user_seq, test_lines, max_item, train_matrix = get_user_seqs(
        train, test
    )

    # 음식점 종류를 저장합니다.
    args.item_list = train['rest_code'].unique()

    # item id의 가장 큰 값 저장합니다.
    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    
    # save model args, (model_name : Finetune_full, data_name : Ml, output_dir : output/)
    args_str = f"{args.model_name}_{args.data_type}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    # user * item 메트릭스.
    args.train_matrix = train_matrix 

    # 모델 기록용 파일 경로 저장합니다.
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    
    # SASRecDataset 클래스를 불러옵니다. (datasets.py 내 존재)
    # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열, => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
    # output : user_id(유저번호), input_ids(item), target_pos(item), target_neg(item), answer(test_item)
    train_dataset = SASRecTrainDataset(args, user_seq)

    # RandomSampler : 데이터 셋을 랜덤하게 섞어줍니다. 인덱스를 반환해줍니다.
    train_sampler = RandomSampler(train_dataset)

    # 모델 학습을 하기 위한 데이터 로더를 만듭니다. 랜덤으로 섞고 배치 단위(defalut : 256)로 출력합니다.
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )

    test_dataloader = DataLoader(
        train_dataset, shuffle=False, batch_size=args.batch_size * 2
    )

    # S3RecModel 모델을 불러옵니다. (models.py 내 존재)
    model = S3RecModel(args=args)
    # 모델을 GPU에 실어요.
    model = model.to(args.device)

    for epoch in range(args.epochs):
        iteration(args, epoch, train_dataloader, model)
        if epoch % 5== 4:
            scores, pred_list = test_score(args, epoch, test_dataloader, model, test_lines)
            print("recall_k = ", scores)

    pred_df = pd.DataFrame(pred_list)
    pred_df['pred'] = pred_df.apply(lambda x : [x[i] for i in range(0,20)], axis = 1)
    pred_df = pred_df.reset_index()
    pred_df = pred_df[['index','pred']]
    pred_df.to_csv(f'{args.model_name}_{args.data_type}_test_{args.data_name}.csv', index = False)
    print(personalizeion(pred_df))
    
    torch.save(model.state_dict(), args.checkpoint_path)
    


if __name__ == "__main__":
    main()
