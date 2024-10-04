import torch.nn as nn
import torch.nn.functional as F
import torch

class TopologicalLoss(nn.Module):
    def __init__(self):
        super(TopologicalLoss, self).__init__()
        
        self.kernel01 = torch.tensor([[1, 1, 1],
                                      [0, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel02 = torch.tensor([[1, 1, 1],
                                      [1, 0, 1],
                                      [1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel03 = torch.tensor([[1, 1, 1],
                                      [1, 0, 0],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel04 = torch.tensor([[1, 0, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel05 = torch.tensor([[0, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel06 = torch.tensor([[1, 1, 1],
                                      [1, 0, 1],
                                      [0, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel07 = torch.tensor([[1, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel08 = torch.tensor([[1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel09 = torch.tensor([ [1, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel20 = torch.tensor([[1, 1, 1, 1, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel21 = torch.tensor([[0, 0, 0, 1, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                     [ 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel22 = torch.tensor([[1, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1],
                                     [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel23 = torch.tensor([[1, 1, 1, 1, 1],
                                     [1, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [1, 1, 0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel24 = torch.tensor([[1, 1, 1, 1, 1],
                                     [1, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel3 = torch.Tensor(  [[0, 0, 1, 1], 
                       [0, 0, 0, 1], 
                       [1, 0, 0, 1],
                       [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel4 = torch.Tensor(  [[1, 1, 0, 0], 
                              [1, 0, 0, 0], 
                              [1, 0, 0, 1],
                              [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel5 = torch.Tensor(  [[1, 1, 1, 1], 
                              [1, 0, 0, 1], 
                              [0, 0, 0, 1],
                              [0, 0, 1, 1]]).view(1,1,4,4)

        self.kernel6 = torch.Tensor(  [[1, 1, 1, 1], 
                              [1, 0, 0, 1], 
                              [1, 0, 0, 0],
                              [1, 1, 0, 0]]).view(1,1,4,4)

        self.kernel7 = torch.Tensor(  [[1, 0, 0, 1], 
                              [1, 0, 0, 1], 
                              [1, 0, 0, 1],
                              [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel8 = torch.Tensor(  [[1, 1, 1, 1], 
                              [0, 0, 0, 1], 
                              [0, 0, 0, 1],
                              [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel9 = torch.Tensor(  [[1, 1, 1, 1], 
                              [1, 0, 0, 1], 
                              [1, 0, 0, 1],
                              [1, 0, 0, 1]]).view(1,1,4,4)

        self.kernel10 = torch.Tensor(  [[1, 1, 1, 1], 
                              [1, 0, 0, 0], 
                              [1, 0, 0, 0],
                              [1, 1, 1, 1]]).view(1,1,4,4)
        self.kernels = [self.kernel01, self.kernel02, self.kernel03, self.kernel04, self.kernel05, self.kernel06, self.kernel07, self.kernel08]

    def soft_threshold(self, x, threshold, alpha=10.0):
        return torch.sigmoid(alpha * (-x + threshold))

    def endpoints(self, mask, kernel, padded=True):
        kernel_size = kernel.shape[-1]
        pad = 2
        if kernel_size == 3:
            pad = 1
        elif kernel_size == 5:
            pad = 2
        neighbors_count = F.conv2d(mask, kernel, padding=pad)
        if kernel_size == 4:
            line_ends = self.soft_threshold(neighbors_count[:,:,:-1, :-1], 1.0) * mask
        else:
            line_ends = self.soft_threshold(neighbors_count, 1.0) * mask
        if padded:
            line_ends = line_ends[:, :, 1:-1, 1:-1]
        return line_ends

    def forward(self, mask):
        epss = torch.zeros_like(mask)
        for kernel in self.kernels:
            if mask.is_cuda:
                kernel = kernel.cuda()
            padding = (1, 1, 1, 1)
            padded_mask = F.pad(mask[0], padding, 'constant', 1)
            eps = self.endpoints(padded_mask.unsqueeze(0), kernel = kernel, padded=True)
            epss += eps
        loss = (epss[0]/len(self.kernels)).sum()
        return loss

class CustSigmoid(nn.Module):
    def __init__(self, alpha=10, beta=0.5):
        super(CustSigmoid, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, x):
        return 1 / (1 + torch.exp(-self.alpha * (x + self.beta)))

