import torch

def my_norm(x):
    return torch.sqrt(torch.sum(x**2))

# now numerically find the gradient
x11 = torch.tensor([1.0,1.0], requires_grad=True)
print('||{}|| = {}'.format(x11.detach().numpy(),(my_norm(x11)).item()))

x00 = torch.tensor([0.0,0.0], requires_grad=True)
print('||{}|| = {}'.format(x00.detach().numpy(),(my_norm(x00)).item()))

f1 = my_norm(x11)
f1.backward()
print('grad_x||{}||={}'.format(x11.detach().numpy(), x11.grad))
x11.grad.zero_()

f1_pt = torch.norm(x11)
f1_pt.backward()
print('torch grad_x||{}||={}'.format(x11.detach().numpy(), x11.grad))

f0 = my_norm(x00)
f0.backward()
print('grad_x||{}||={}'.format(x00.detach().numpy(), x00.grad))
x00.grad.zero_()

f0_pt = torch.norm(x00)
f0_pt.backward()
print('torch grad_x||{}||={}'.format(x00.detach().numpy(), x00.grad))


print('Hello')