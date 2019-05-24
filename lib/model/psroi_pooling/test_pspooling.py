import torch
from modules.psroi_pool import PSRoIPool
from torch.autograd import gradcheck

if __name__=='__main__':
    input_size = (2,7*7*10,30,30)
    input = torch.rand(input_size).cuda()
    rois = torch.tensor([[0,10,10,18,18],[1,5,5,10,10]]).cuda()
    save = True
    #############
    print(torch.autograd.gradcheck(PSRoIPool(), (input, rois)))
    #############
    res = gradcheck(PSRoIPool(), (input, rois), raise_exception=False)
    print(res)