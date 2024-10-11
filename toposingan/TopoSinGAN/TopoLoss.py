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
        
        
        self.kernel1 = torch.Tensor(  [[0, 0, 1, 1], 
                                       [0, 0, 0, 1], 
                                       [1, 0, 0, 1],
                                       [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel2 = torch.Tensor(  [[1, 1, 0, 0], 
                                      [1, 0, 0, 0], 
                                      [1, 0, 0, 1],
                                      [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel3 = torch.Tensor(  [[1, 1, 1, 1], 
                                      [1, 0, 0, 1], 
                                      [0, 0, 0, 1],
                                      [0, 0, 1, 1]]).view(1,1,4,4)

        self.kernel4 = torch.Tensor(  [[1, 1, 1, 1], 
                                      [1, 0, 0, 1], 
                                      [1, 0, 0, 0],
                                      [1, 1, 0, 0]]).view(1,1,4,4)

        self.kernel5 = torch.Tensor(  [[1, 0, 0, 1], 
                                      [1, 0, 0, 1], 
                                      [1, 0, 0, 1],
                                      [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel6 = torch.Tensor(  [[1, 1, 1, 1], 
                                      [0, 0, 0, 1], 
                                      [0, 0, 0, 1],
                                      [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel7 = torch.Tensor(  [[1, 1, 1, 1], 
                                      [1, 0, 0, 1], 
                                      [1, 0, 0, 1],
                                      [1, 0, 0, 1]]).view(1,1,4,4)

        self.kernel8 = torch.Tensor(  [[1, 1, 1, 1], 
                                      [1, 0, 0, 0], 
                                      [1, 0, 0, 0],
                                      [1, 1, 1, 1]]).view(1,1,4,4)

        
        self.kernel10 = torch.tensor([[0, 0, 0, 1, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel20 = torch.tensor([[1, 1, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 1],
                                      [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel30 = torch.tensor([[1, 1, 1, 1, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [1, 1, 0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel40 = torch.tensor([[1, 1, 1, 1, 1],
                                      [1, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel50 = torch.tensor([[1, 1, 1, 1, 1],
                                      [1, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel60 = torch.tensor([[1, 1, 1, 1, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel70 = torch.tensor([[1, 1, 1, 1, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel80 = torch.tensor([[1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.kernel100 = torch.tensor([[0, 0, 0, 0, 1, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel200 = torch.tensor([[1, 1, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel300 = torch.tensor([[1, 1, 1, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel400 = torch.tensor([[1, 1, 1, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel500 = torch.tensor([[1, 1, 1, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel600 = torch.tensor([[1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [1, 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel700 = torch.tensor([[1, 1, 1, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel800 = torch.tensor([[1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        
        
        self.kernels = [
            self.kernel01, self.kernel02, self.kernel03, self.kernel04, self.kernel05, self.kernel06, self.kernel07, self.kernel08, 
            self.kernel1, self.kernel2, self.kernel3, self.kernel4, self.kernel5, self.kernel6, self.kernel7, self.kernel8, 
            self.kernel10, self.kernel20, self.kernel30, self.kernel40, self.kernel50, self.kernel60, self.kernel70, self.kernel80, 
            self.kernel100, self.kernel200, self.kernel300, self.kernel400, self.kernel500, self.kernel600, self.kernel700, self.kernel800]

    def soft_threshold(self, x, beta, alpha=10.0):
        return torch.sigmoid(alpha * (-x + beta))

    def endpoints(self, mask, kernel, padded=True):
        kernel_size = kernel.shape[-1]
        pad = kernel_size // 2
        neighbors_count = F.conv2d(mask, kernel, padding=pad)
        if neighbors_count.shape != mask.shape:
            neighbors_count = neighbors_count[:, :, :mask.shape[2], :mask.shape[3]]
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
        tcp_map = self.soft_threshold(epss[0], alpha=10.0, beta = epss[0].mean())
        loss = (epss[0]/len(self.kernels)).sum()
        return [loss, [epss, mask]]

class CustSigmoid(nn.Module):
    def __init__(self, alpha=10, beta=0.5):
        super(CustSigmoid, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, x):
        return 1 / (1 + torch.exp(-self.alpha * (x + self.beta)))
