import torch
from torch.profiler import profile, record_function, ProfilerActivity
from connectfour_tensor import C4NN
from tictactoe_tensor import TTTNN

tttm = TTTNN()
c4m = C4NN()

tttx = torch.randn(10, 2, 3, 3)
c4x = torch.randn(10, 2, 7, 6)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        c4m(c4x)


print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))



