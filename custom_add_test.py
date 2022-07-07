import torch
from _mps_test import custom_add

a = torch.rand(3, 5).to('mps')
b = torch.rand(3, 5).to('mps')

gt = a + b
out = custom_add(a, b)

torch.testing.assert_close(out, gt)

print('finish')
