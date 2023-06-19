import torch
import torch.nn.functional as F
import torch.nn as nn

class ADMMSolverBlock(nn.Module):
    def __init__(self,rho,lamb,eta,grad="1st",T=1):
        super(ADMMSolverBlock, self).__init__()
        # params
        self.T = T
        self.grad = grad
        # variables
        self.beta = None
        self.Q = None
        self.count = 0
        # blocks
        self.get_gradients = Sobel()
        self.apply_threshold = SoftThresholding(rho,lamb)
        self.update_multipliers = MultiplierUpdate(eta)

    def forward(self, F, masks, vox_dim):
        # get masked grads
        dF = self.get_gradients(F, vox_dim) #[dF/dx, dF/dy, dF/dz]
        
        c = [df * mask for df, mask in zip(dF, masks)]

        c = torch.cat(c, dim = 1) #[B,4,H,W]
        # initialize 
        beta = torch.zeros_like(c)
        q = torch.zeros_like(c)
        
        Q = [q]
        C = [c]
        Betas = [beta]

        # update q and beta
        for t in range(self.T):
            q = self.apply_threshold(c,beta,t)
            beta = self.update_multipliers(q,c,beta)

            Q.append(q)
            C.append(c)
            Betas.append(beta)

        #return [Q[-1]], [C[-1]], [Betas[-1]]
        self.count += 1
        return Q, C, Betas
    
class SoftThresholding(nn.Module):
    def __init__(self,rho,lamb):
        super(SoftThresholding, self).__init__()
        if type(lamb) is list: # support several lambda values
            self.lamb = lamb
        else:
            self.lamb = [lamb]
        self.rho = rho
    
    def forward(self,C, beta, i=0):
        th = self.lamb[i] / self.rho

        mask = (C - beta).abs() >= th
        Q = (C - beta - th * torch.sign(C - beta)) * mask
        
        return Q

class MultiplierUpdate(nn.Module):
    def __init__(self, eta):
        super(MultiplierUpdate,self).__init__()
        self.eta = eta

    def forward(self, Q, C, beta):
        beta = beta + self.eta * (Q - C)
        
        return beta

class MaskGenerator(nn.Module):
    def __init__(self,alpha,learn_mask=False):
        super(MaskGenerator,self).__init__()
        self.learn_mask = learn_mask
        self.alpha = alpha
        self.sobel = Sobel()

    def forward(self, image, vox_dim, scale=1/8):
        if self.learn_mask: 
            im_grads = self.sobel(image,vox_dim) #[dx, dy, dz]
            encoders = [self.ddx_encoder, self.ddy_encoder, self.ddz_encoder]
            masks = [enc(grad.abs()) for enc, grad in zip(encoders, im_grads)]
        else:
            image = F.interpolate(image, scale_factor=scale, mode='trilinear')
            im_grads = self.sobel(image,vox_dim) #[dx, dy, dz]

            masks = [torch.exp(-torch.mean(torch.abs(grad), 1, keepdim=True) * self.alpha) for grad in im_grads]

        return masks

class Sobel(nn.Module):
    def __init__(self,  f_x = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                               [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                        f_y = [[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                               [[1, 2, 1], [2, 4, 2], [1, 2, 1]]],
                        f_z = [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                               [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]]):
        super(Sobel, self).__init__()
        Dx = torch.tensor(f_x, dtype = torch.float, requires_grad = False).view(1,1,3,3,3)
        Dy = torch.tensor(f_y, dtype = torch.float, requires_grad = False).view(1,1,3,3,3)
        Dz = torch.tensor(f_z, dtype = torch.float, requires_grad = False).view(1,1,3,3,3)

        self.D = nn.Parameter(torch.cat((Dx, Dy, Dz), dim=0), requires_grad=False)
    
    def forward(self, image, vox_dims):
        # apply filter over each channel seperately
        im_ch = torch.split(image, 1, dim = 1)
        grad_ch = [F.conv3d(ch, self.D, padding = 1) for ch in im_ch]
        #grad = F.conv3d(image, self.D, padding=1)

        dx = torch.cat([g[:,0:1,:,:] for g in grad_ch], dim=1)
        dy = torch.cat([g[:,1:2,:,:] for g in grad_ch], dim=1)
        dz = torch.cat([g[:,2:3,:,:] for g in grad_ch], dim=1)

        #dx = grad[:,0:1,:,:,:]
        #dy = grad[:,1:2,:,:,:]
        #dz = grad[:,2:3,:,:,:]

        return [dl / l for dl,l in zip([dx, dy, dz],vox_dims.squeeze())]
