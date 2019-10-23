from DSConv.nn.bfp_quantization import BFPActivationLegacy
from src.bfpactivation import BFPActivation
import torch
import numpy as np
from os import environ as env

from prettytable import PrettyTable

env["CUDA_VISIBLE_DEVICES"] = "1"
env["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == "__main__":

    func_act = BFPActivationLegacy(4, 7, 32)
    theo_activation = BFPActivation(4, 7, 32)

    number_blocks = 3
    batch_size = 2
    block_size = 32
    width = 3
    height = 3

    act = torch.randn((batch_size, block_size*number_blocks, width, height))

    sol = func_act(act)
    sol_theo = theo_activation(act).cpu()

    x = PrettyTable()
    col_names = ["orig", "cpp", "pytorch", "error"]
    x.add_column(col_names[0], act[0, :, 0, 0].numpy())
    x.add_column(col_names[1], sol_theo[0, :, 0, 0].numpy())
    x.add_column(col_names[2], sol[0, :, 0, 0].numpy())
    x.add_column(
        col_names[3], sol_theo[0, :, 0, 0].numpy() - sol[0, :, 0, 0].numpy()
    )

    print(x)

    # print("Original:", act[0, 0, :, 0, 0])
    # print("Theo's act:", sol_theo[0, 0, :, 0, 0])
    # print("Mine:", sol[0, 0, :, 0, 0])
    # print(sol_theo[0, 0, :, 0, 0] - sol[0, 0, :, 0, 0])

    # print(sol)
    # print(sol_theo)
    # print(f"All close? :{np.allclose(sol, sol_theo)}")
    # print(sol_theo - sol)
