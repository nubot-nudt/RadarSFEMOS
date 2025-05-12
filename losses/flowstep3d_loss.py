import torch
from torch.nn import Module, MSELoss, L1Loss
from lib import pointnet2_utils_step as pointutils

def FlowStep3D_self_loss(pos1, pos2, flows_pred):

    loss_iters_w = [0.8, 0.2, 0.4, 0.6]
    w_data = [0.75, 0.75, 0.75, 0.75]
    w_smoothness = [0.25, 0.25, 0.25, 0.25]
    smoothness_loss_params = {
        'w_knn': 3, 
        'w_ball_q': 1,
        'knn_loss_params': {
            'k': 16,
            'radius': 0.25,
            'loss_norm': 1,
            },
        'ball_q_loss_params':{
            'k': 64,
            'radius': 0.75,
            'loss_norm': 1
            }
    }
    chamfer_loss_params ={
        'loss_norm': 2,
        'k': 1
    }
        

    loss_obj = UnSupervisedL1Loss(w_data, w_smoothness, smoothness_loss_params, chamfer_loss_params)
    loss = 0
    for i, w in enumerate(loss_iters_w):
        loss += w * loss_obj(pos1.transpose(2,1).contiguous(), pos2.transpose(2,1).contiguous(), flows_pred[i], None, i)
    
    items={
        'Loss': loss.item(),
        }
    return loss, items


def FlowStep3D_sv_loss(pos1, pos2, flows_pred, flow_gt):

    loss_iters_w = [0.7, 0.4, 0.4, 0.4]
    w_data = [0.9, 0.99, 0.99, 0.99]
    w_smoothness = [0.1, 0.01, 0.01, 0.01]
    smoothness_loss_params = {
        'w_knn': 3, 
        'w_ball_q': 1,
        'knn_loss_params': {
            'k': 16,
            'radius': 0.5,
            'loss_norm': 1,
            },
        'ball_q_loss_params':{
            'k': 24,
            'radius': 0.5,
            'loss_norm': 1
        }
    }
    loss_obj = SupervisedL1RegLoss(w_data, w_smoothness, smoothness_loss_params)
    loss = 0
    for i, w in enumerate(loss_iters_w):
        loss += w * loss_obj(pos1.transpose(2,1).contiguous(), pos2.transpose(2,1).contiguous(), flows_pred[i], flow_gt, i)
    
    items={
        'Loss': loss.item(),
        }
    return loss, items


class SupervisedL1RegLoss(Module):
    def __init__(self, w_data, w_smoothness, smoothness_loss_params, **kwargs):
        super(SupervisedL1RegLoss, self).__init__()
        self.data_loss = L1Loss()
        self.smoothness_loss = SmoothnessLoss(**smoothness_loss_params)
        self.w_data = w_data
        self.w_smoothness = w_smoothness

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor, i=0) -> torch.Tensor:
        if len(self.w_data) == 1:
            w_data = self.w_data[0]
            w_smoothness = self.w_smoothness[0]
        else:
            w_data = self.w_data[i]
            w_smoothness = self.w_smoothness[i]

        loss = (w_data * self.data_loss(pred_flow, gt_flow)) + (w_smoothness * self.smoothness_loss(pc_source, pred_flow)) 

        return loss


class KnnLoss(Module):
    def __init__(self, k, radius, loss_norm, **kwargs):
        super(KnnLoss, self).__init__()
        self.k = k
        self.radius = radius
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        flow = pred_flow.permute(0, 2, 1).contiguous()
        dist, idx = pointutils.knn(self.k, pc_source, pc_source)
        tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, self.k).to(idx.device)
        idx[dist > self.radius] = tmp_idx[dist > self.radius]
        nn_flow = pointutils.grouping_operation(flow, idx.detach())
        loss = (flow.unsqueeze(3) - nn_flow).norm(p=self.loss_norm, dim=1).mean(dim=-1)
        return loss.mean()


class BallQLoss(Module):
    def __init__(self, k, radius, loss_norm, **kwargs):
        super(BallQLoss, self).__init__()
        self.k = k
        self.radius = radius
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        flow = pred_flow.permute(0, 2, 1).contiguous()
        idx = pointutils.ball_query(self.radius, self.k, pc_source, pc_source)
        nn_flow = pointutils.grouping_operation(flow, idx.detach())  # retrieve flow of nn
        loss = (flow.unsqueeze(3) - nn_flow).norm(p=self.loss_norm, dim=1).mean( dim=-1)
        return loss.mean()


class SmoothnessLoss(Module):
    def __init__(self, w_knn, w_ball_q, knn_loss_params, ball_q_loss_params, **kwargs):
        super(SmoothnessLoss, self).__init__()
        self.knn_loss = KnnLoss(**knn_loss_params)
        self.ball_q_loss = BallQLoss(**ball_q_loss_params)
        self.w_knn = w_knn
        self.w_ball_q = w_ball_q

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        loss = (self.w_knn * self.knn_loss(pc_source, pred_flow)) + (self.w_ball_q * self.ball_q_loss(pc_source, pred_flow))
        return loss


class ChamferLoss(Module):
    def __init__(self, k, loss_norm, **kwargs):
        super(ChamferLoss, self).__init__()
        self.k = k
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        pc_target = pc_target.contiguous()
        pc_target_t = pc_target.permute(0, 2, 1).contiguous()
        pc_pred = (pc_source + pred_flow).contiguous()
        pc_pred_t = pc_pred.permute(0, 2, 1).contiguous()

        _, idx = pointutils.knn(self.k, pc_pred, pc_target)
        nn1 = pointutils.grouping_operation(pc_target_t, idx.detach())
        dist1 = (pc_pred_t.unsqueeze(3) - nn1).norm(p=self.loss_norm, dim=1).mean( dim=-1)  # nn flow consistency
        _, idx = pointutils.knn(self.k, pc_target, pc_pred)
        nn2 = pointutils.grouping_operation(pc_pred_t, idx.detach())
        dist2 = (pc_target_t.unsqueeze(3) - nn2).norm(p=self.loss_norm, dim=1).mean( dim=-1)  # nn flow consistency
        ch_dist = (dist1 + dist2)
        return ch_dist.mean()


class UnSupervisedL1Loss(Module):
    def __init__(self, w_data, w_smoothness, smoothness_loss_params, chamfer_loss_params, **kwargs):
        super(UnSupervisedL1Loss, self).__init__()
        self.data_loss = ChamferLoss(**chamfer_loss_params)
        self.smoothness_loss = SmoothnessLoss(**smoothness_loss_params)
        self.w_data = w_data
        self.w_smoothness = w_smoothness

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor, i=0) -> torch.Tensor:
        if len(self.w_data) == 1:
            w_data = self.w_data[0]
            w_smoothness = self.w_smoothness[0]
        else:
            w_data = self.w_data[i]
            w_smoothness = self.w_smoothness[i]
        loss = (w_data * self.data_loss(pc_source, pc_target, pred_flow)) + (w_smoothness * self.smoothness_loss(pc_source, pred_flow))
        return loss