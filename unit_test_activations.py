from DSConv.nn.bfp_quantization import BFPQuant
from src.bfpactivation import BFPActivation
import torch
from os import environ as env

from prettytable import PrettyTable
import timeit

env["CUDA_VISIBLE_DEVICES"] = "1"
env["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == "__main__":

    m_bits = 3
    func_act = BFPQuant.apply
    theo_activation = BFPActivation(m_bits)

    number_blocks = 12
    batch_size = 256
    block_size = 32
    width = 3
    height = 3

    act = 2*torch.randn((number_blocks, batch_size, block_size, width, height))

    sol = func_act(act, -127, 128, m_bits, -(2**m_bits-1), (2**m_bits -1))
    sol_theo = theo_activation(act).cpu()

    diff = sol - sol_theo

    print(torch.nonzero(diff))
    
    if not torch.nonzero(diff).size()[0]:
        print("The Difference is Zero")
    else:
        indeces = []
        for idx in torch.nonzero(diff):
            indeces.append((idx[0].item(), idx[1].item(), idx[3].item(), idx[4].item()))

            x = PrettyTable()
            col_names = ["orig", "cpp", "pytorch", "error"]
            x.add_column(col_names[0], act[idx[0], idx[1], :, idx[3], idx[4]].numpy())
            x.add_column(col_names[1], sol_theo[idx[0], idx[1], :, idx[3], idx[4]].numpy())
            x.add_column(col_names[2], sol[idx[0], idx[1], :, idx[3], idx[4]].numpy())
            x.add_column(
                col_names[3], sol_theo[idx[0], idx[1], :, idx[3], idx[4]].numpy() - sol[idx[0], idx[1], :, idx[3], idx[4]].numpy()
            )

            print(x)
            
    setup_cpu = '''import torch
from src.bfpactivation import BFPActivation

number_blocks = 256
batch_size = 256
block_size = 32
width = 3
height = 3

m_bits = 3
theo_activation = BFPActivation(m_bits)
act = 2*torch.randn((number_blocks, batch_size, block_size, width, height))
act = act.cuda()
    '''

    setup_cpu_legacy = '''import torch
from DSConv.nn.Activation import BFPQuant

number_blocks = 256
batch_size = 256
block_size = 32
width = 3
height = 3

m_bits = 3
func_act = BFPQuant.apply
act = 2*torch.randn((number_blocks, batch_size, block_size, width, height))
act = act.cuda()
    '''
    timing = timeit.repeat("theo_activation(act)", setup=setup_cpu,  number=100)
    print("Theo Timing", timing)
    timing = timeit.repeat("func_act(act, -127, 128, m_bits, -(2**m_bits-1), (2**m_bits -1))", setup=setup_cpu_legacy,  number=100)
    print("Legacy Timing", timing)