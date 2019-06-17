import torch
from torch import nn
from model.utils.config import cfg

class trNMS(nn.Module):
    def __init__(self,
            PANELTY_K = cfg.SIAMESE.PANELTY_K,
            HANNING_WINDOW_WEIGHT = cfg.SIAMESE.HANNING_WINDOW_WEIGHT,
            HANNING_WINDOW_SIZE_FACTOR = cfg.SIAMESE.HANNING_WINDOW_SIZE_FACTOR):
        super(trNMS, self).__init__()
        self.PANELTY_K = PANELTY_K
        self.HANNING_WINDOW_WEIGHT = HANNING_WINDOW_WEIGHT
        self.HANNING_WINDOW_SIZE_FACTOR = HANNING_WINDOW_SIZE_FACTOR

    def forward(self, rois, rpn_rois, scores):
        return self.nms(rois, rpn_rois, scores)

    def nms(self, rois, rpn_rois, scores):
        '''
        This function is to do non-maximum suppression to select the best tracked target for each roi.
        :param rois: rois used for tracking. (N, 4) 4 represents coordinates tuple of x1,y1,x2,y2.
        :param rpn_rois: detected rois in a new frame for all rois. (N, n, 4) n is 2000 for training and 300 for testing by default.
        :param scores: the scores for all rois. (N, n, 1).
        :return: boxes and scores of the selected rois. (N, 4) and (N, 1).
        '''
        rois = rois.unsqueeze(1) # expand dims to (N,1,4).
        x,y,w,h,s,r = self.get_rois_vals(rois)
        x_,y_,w_,h_,s_,r_ = self.get_rois_vals(rpn_rois[:,:,1:5])
        sz = s.size()

        s_max = torch.max(s/s_, s_/s)
        r_max = torch.max(r, 1.0/r)
        penalty_score = ((-self.PANELTY_K)*(s_max*r_max-1.0)).exp()

        window_sz = s*self.HANNING_WINDOW_SIZE_FACTOR
        dist = ((x-x_).pow(2)+(y-y_).pow(2)).sqrt()
        hanning_score = 0.5+0.5*((dist*3.141592653589793/window_sz).cos())
        #print('dist:',dist)
        #print('hanning_score:',hanning_score)
        hn_sz = hanning_score.size()
        hanning_score = hanning_score.view(-1)

        zero_inds = torch.nonzero((dist>window_sz).view(-1))
        if zero_inds.size(0)>0:
            zero_inds = zero_inds[:,0]
            hanning_score[zero_inds] = 0
        hanning_score = hanning_score.view(hn_sz)
        #print('scores:')
        #print('penalty_score:')#, penalty_score)
        #print('hanning_score:')#, hanning_score)
        #for a,b,c in zip(scores[:,:,1].view(-1,1), penalty_score.view(-1,1), hanning_score.view(-1,1)):
        #    print(a,',',b,',',c)
        # TODO change back.
        penalty_window = scores[:,:,1]*penalty_score+hanning_score*self.HANNING_WINDOW_WEIGHT#+penalty_score
        #penalty_window = scores[:,:,1]
        #penalty_window = hanning_score
        #penalty_window = penalty_score
        # inds should be of size N.
        inds = penalty_window.argmax(dim=1)

        sel_rpn_rois = rpn_rois.new(rpn_rois.size(0), 4).zero_()
        sel_scores = scores.new(scores.size(0), 1).zero_()
        for i in range(rpn_rois.size(0)):
            sel_rpn_rois[i,:] = rpn_rois[i,inds[i],1:]
            sel_scores[i,:] = scores[i,inds[i],1:]
        return sel_rpn_rois, sel_scores

    def get_rois_vals(self, rois):
        x = (rois[:, :, 0] + rois[:, :, 2])/2.0
        y = (rois[:, :, 1] + rois[:, :, 3])/2.0
        w = (rois[:, :, 0] - rois[:, :, 2]).abs()+1e-4
        h = (rois[:, :, 1] - rois[:, :, 3]).abs()+1e-4
        p = (w + h) / 2
        s = ((w + p) * (h + p)).sqrt()
        r = w / h
        return x,y,w,h,s,r