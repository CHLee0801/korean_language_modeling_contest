import torch
output = torch.tensor([[0.3, 0.1, 0.4, 0]])

print(output)
if sum(torch.where(output > 0.5, 1, 0)[0]) == 0:
    output[0][torch.argmax(output[0])] = 1
    print(output[0])