import numpy as np
import torch
from .model import MultiVAE


def recommend(rest_codes, K=20):
    user_list = eval(rest_codes)
    zero_list = torch.tensor(np.zeros(41460)).float()
    for i in user_list:
        zero_list[i] = 1
    zero_list =zero_list.unsqueeze(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MultiVAE(p_dims=[100, 400, 41460])
    model.load_state_dict(torch.load(f'/opt/ml/input/project/backend/app/models/data/multivae-time.pt'))
    model.to(device)
    model.eval()

    with torch.no_grad():
        mat = zero_list
        mat = mat.to(device)

        recon_mat = model(mat)
        recon_mat[mat == 1] = -np.inf
        rec_list = recon_mat.argsort(dim=1)

        for rec in rec_list:
            up = rec[-K:].cpu().numpy().tolist()

    return up


if __name__ == '__main__':
    user_list = '[88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77]'
    ans = recommend(user_list)
