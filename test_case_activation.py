from DSConv.nn.Activation import BFPQuant
from src.bfpactivation import BFPActivationFunction
import torch
import numpy as np

from prettytable import PrettyTable

if __name__ == "__main__":

    func_act = BFPQuant.apply
    theo_activation = BFPActivationFunction.apply

    number_blocks = 1
    batch_size = 1
    block_size = 32
    width = 1
    height = 1

    act = torch.randn((number_blocks, batch_size, block_size, width, height))

    sol = func_act(act, -127, 128, 3, -7, 7)
    sol_theo = theo_activation(act, 3)

    x = PrettyTable()
    col_names = ["orig", "cpp", "pytorch", "error"]
    x.add_column(col_names[0], act[0, 0, :, 0, 0].numpy())
    x.add_column(col_names[1], sol_theo[0, 0, :, 0, 0].numpy())
    x.add_column(col_names[2], sol[0, 0, :, 0, 0].numpy())
    x.add_column(
        col_names[3], sol_theo[0, 0, :, 0, 0].numpy() - sol[0, 0, :, 0, 0].numpy()
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