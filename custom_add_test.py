import torch
from _mps_test import custom_add

a = torch.rand(3, 5).to('mps')
b = torch.rand(3, 5).to('mps')

torch.testing.assert_close(custom_add(a, b), a + b)

print('finish')
