import torch
from torch.utils.data import Dataset

from utils import neg_sample


class SASRecTrainDataset(Dataset):
    def __init__(self, args, user_seq, test_user_seq):
        # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열, => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        self.args = args
        self.user_seq = user_seq
        self.test_user_seq = test_user_seq
        self.max_len = args.max_seq_length
        self.part_sequence = []


    def __getitem__(self, index):
        """_summary_
        Returns:
            user_id : 유저 번호(정렬 되있음 = index와 같음)
            input_ids : 해당 유저의 test 음식점[:-1]
            target_pos : 해당 유저의 train 음식점[1:] 정답지.
            target_neg : 네거티브 샘플링
            answer : 해당 유저의 test 음식점 정답지.
        """        
        # sequence : part_sequence의 해당 index에 저장된 sequence
        sequence = self.user_seq[index]  # pos_items
        input_ids = sequence[:-1]
        target_pos = sequence[1:]
        target_neg = []
        # Test도 안걸리기 때문에, Test 환경과 동일하게 맞춰주기 위해
        user_set = set(sequence)

        for _ in input_ids:
            target_neg.append(neg_sample(user_set, self.args.item_size))

        # padding
        # max_len 길이에 맞춰서 앞쪽 0으로 채움
        # pad_len 값이 음수이면, [0] * pad_len = []
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        # 길이 max_len으로 통일
        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        # check length
        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len
        
        user_id = index
        answer = self.test_user_seq[index] # test
        # to tensor
        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long), # input_ids 대비 하나씩 밀림.
            torch.tensor(target_neg, dtype=torch.long), # input_ids 길이만큼 네거티브 샘플링.
            torch.tensor(answer, dtype=torch.long)
        )

        return cur_tensors

    def __len__(self):
        # user 수 반환
        return len(self.user_seq)



class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="submission"):
        # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열, => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):
        # index로 user_id 사용
        # user_id를 index로 사용
        user_id = index
        items = self.user_seq[index]

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4, 5]
        # target [1, 2, 3, 4, 5, 6]
        # answer [6]

        # submission [0, 1, 2, 3, 4, 5, 6]
        # answer None

        input_ids = items[:]
        target_pos = items[:]  # will not be used
        answer = []

        target_neg = []
        seq_set = set(items)
        # input_ids 길이만큼 target_neg에 negative samples 생성
        # 자세한건 neg_sample 함수 내에 써놨습니다.
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        # padding
        # max_len 길이에 맞춰서 앞쪽 0으로 채움
        # pad_len 값이 음수이면, [0] * pad_len = []
        pad_len = self.max_len - len(input_ids)
        breakpoint()
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        # 길이 max_len으로 통일
        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        # check length
        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        # to tensor
        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long), # input_ids 대비 하나씩 밀림.
            torch.tensor(target_neg, dtype=torch.long), # input_ids 길이만큼 네거티브 샘플링.
            torch.tensor(answer, dtype=torch.long), # 마지막 값.
        )
        return cur_tensors

    def __len__(self):
        # user 수 반환
        return len(self.user_seq)
