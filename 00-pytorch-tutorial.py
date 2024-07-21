import torch


def analys_tensors(tensor):
    scalar= torch.tensor(tensor)
    print('tensor is: ',scalar)

    print('ndim is: ',scalar.ndim)
    print('size is: ',scalar.size())
    print('shape is: ',scalar.shape)
    print("*"*10)

analys_tensors(3)
analys_tensors([1,2,3,4,5,6])
analys_tensors([[1,2,3,4],[1,2,3,4]])
analys_tensors(3)
analys_tensors(3)

# Create a tensor of all zeros
zeros = torch.zeros(size=(4, 4,3))
zeros, zeros.dtype

newTensor_flt = torch.arange(0,4,0.5).reshape(2,4)
print(newTensor_flt)
newTensor_flt.ndim
newTensor_flt.size()
newTensor_flt.dtype
newTensor_flt.device

newTensor_int = torch.arange(0,8,1).reshape(2,4)
print(newTensor_int)

newTensor_int.dtype = torch.float16
newTensor_int.ndim
newTensor_int.size()
newTensor_int.dtype
newTensor_int.device

newTensor_int = newTensor_int.to(dtype=torch.float32)

device = torch.device()
newTensor_flt = newTensor_flt.reshape(4,2)
newTensor_int.matmul(newTensor_flt)

newTensor_flt = newTensor_flt.reshape(1,-1)
newTensor_flt.squeeze()

newTensor_flt = torch.tensor([[1,2,3],[3,4,5],[-1,0,-1]])
newTensor_flt_new = torch.tensor([[3,0,3],[3,4,5],[-1,0,-1]])

newTensor_flt.mm(newTensor_flt.T) 

torch.range(0, 100, 10)
newTensor_flt_new.max()
newTensor_flt_new.argmax()

newTensor_flt_new[-1][-1]

newTensor_flt_new.sum()
newTensor_flt_new.to(dtype=torch.float32).mean()
newTensor_flt_new.dtype
newTensor_flt_new.to(dtype=torch.float32).element_size() * newTensor_flt_new.numel()


tensor_stack = torch.stack([newTensor_flt,newTensor_flt_new],dim=0)
tensor_stack
tensor_squeeze = tensor_stack.squeeze()

torch.squeeze(tensor_stack)
tensor_stack.size()


import torch

# Create a 2D tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original tensor:\n", tensor)
print("Original shape:", tensor.shape)

# Unsqueeze to add a dimension of size 1 at position 0
unsqueezed_tensor = torch.unsqueeze(tensor, 0)
print("Unsqueezed tensor at dim 0:\n", unsqueezed_tensor)
print("Unsqueezed shape at dim 0:", unsqueezed_tensor.shape)

# Unsqueeze to add a dimension of size 1 at position 1
unsqueezed_tensor = torch.unsqueeze(tensor, 1)
print("Unsqueezed tensor at dim 1:\n", unsqueezed_tensor)
print("Unsqueezed shape at dim 1:", unsqueezed_tensor.shape)

unsqueezed_tensor = torch.unsqueeze(tensor, 2)
print("Unsqueezed tensor at dim 1:\n", unsqueezed_tensor)
print("Unsqueezed shape at dim 1:", unsqueezed_tensor.shape)



