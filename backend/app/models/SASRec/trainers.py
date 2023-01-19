import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from tqdm import tqdm

from utils import ndcg_k, recall_at_k


# PretrainTrainer, FinetuneTrainer 모두 Trainer을 상속받는 클래스.
class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        submission_dataloader,
        args,
    ):

        # 모든 args를 클래스 내 저장합니다.
        self.args = args

        # GPU 상태를 클래스 내 저장합니다.
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        # 입력된 모델을 클래스 내 저장합니다.
        self.model = model
        # 만약 GPU 사용이 가능하다면 모델을 GPU에 실습니다.
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.submission_dataloader = submission_dataloader

        # betas : Adam 옵티마이저 하이퍼 파라미터
        # (참고 : https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        # BCELoss(2진 분류시 일반적으로 사용되는 손실함수)를 클래스에 저장합니다.
        self.criterion = nn.BCELoss()

    # train, valid, test, submission 모두 iteration을 진행하는 함수입니다.
    # iteration은 Trainer에 없어서 상속하는 클래스 Trainer에서 구현해야합니다.
    # iteration은 배치단위로 모델을 학습하거나 예측값을 제출하는 함수입니다.
    # iteration은 현재 FinetuneTrainer(run_train.py)에만 구현되어있습니다.
    # 즉 FinetuneTrainer 이외에는 train, valid, test, submission 함수를 사용하지 않습니다.
    # PretrainTrainer(run_pretrain.py)에서는 pretrain 함수를 대신 사용합니다.
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def submission(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "RECALL@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "RECALL@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
        }
        print(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (
            (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
        )  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred  # [B item_num]


class FinetuneTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        submission_dataloader,
        args,
    ):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            submission_dataloader,
            args,
        )

    def iteration(self, epoch, dataloader, mode="train"):

        # tqdm을 통해 iter를 만듭니다.
        # 핵심은 enumerate(dataloader) 이 것만 기억하셔도 문제 없습니다.
        # 나머지 코드는 tqdm 출력을 이쁘게 도와주는 도구입니다.
        rec_data_iter = tqdm(
            enumerate(dataloader),
            desc="Recommendation EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        self.model.eval()

        pred_list = None
        answer_list = None
        for i, batch in rec_data_iter:

            batch = tuple(t.to(self.device) for t in batch)
            user_ids, input_ids, _, target_neg, answers = batch
            recommend_output = self.model.finetune(input_ids)  # [B L H]

            recommend_output = recommend_output[:, -1, :]  # [B H]

            rating_pred = self.predict_full(recommend_output)  # [B item_num]

            rating_pred = rating_pred.cpu().data.numpy().copy()
            batch_user_index = user_ids.cpu().numpy()
            # 해당 User가 시청한 영화 제외
            rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = -np.inf
            # mask_id 제외
            rating_pred[:, -1] = -np.inf

            # TOP 10 index 추출
            ind = np.argpartition(rating_pred, -10)[:, -10:]
            #ind = np.argpartition(rating_pred, -20)[:, -20:]

            # Top 10 값 추출
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

            # Top 10 역정렬된 순서대로 index 추출
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

            # user 별 Top 10 item index 높은 값 순서대로 10개 추출
            batch_pred_list = ind[
                np.arange(len(rating_pred))[:, None], arr_ind_argsort
            ]

            if i == 0:
                pred_list = batch_pred_list
                answer_list = answers.cpu().data.numpy()
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
                answer_list = np.append(
                    answer_list, answers.cpu().data.numpy(), axis=0
                )

            if mode == "submission":
                return pred_list
            else:
                return self.get_full_sort_score(epoch, answer_list, pred_list)
