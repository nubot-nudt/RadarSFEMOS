import torch
import torch.nn as nn
import sys
from .extractor import FlotEncoder
from .corr import CorrBlock
from .update import UpdateBlock
from .refine import *
from .RadarTransformer import *
from .transformer import *
from utils.model_utils.radarsfemos_util import *
import time
from .pointtransformer_seg import *
from .corr import *
class RSF(nn.Module):
    def __init__(self, args):
        super(RSF, self).__init__()
        self.hidden_dim = 64
        self.context_dim = 64
        self.rigid_pcs = 0.25
        self.rigid_thres = 0.15
        self.feature_extractor = FlotEncoder()
        self.context_extractor = FlotEncoder()

        # self.context_extractor = TransformerBlock1(128)
        # self.pt = TransformerBlock1(128)
        # self.context_extractor = TransformerBlock(128,256,32)
        # self.pt = TransformerBlock(128,256,32)
        args.corr_levels = 3
        args.base_scales = 0.25
        args.truncate_k = int(args.num_points/4)
        self.corr_block = CorrBlock(num_levels=args.corr_levels, base_scale=args.base_scales,
                                    resolution=3, truncate_k=args.truncate_k)
        self.update_block = UpdateBlock(hidden_dim=self.hidden_dim)
        self.num_neighbors = 32
        # self.fc1 = nn.Sequential(
        #     nn.Conv1d(3, 128, kernel_size=1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 128, kernel_size=1, bias=False)
        # )        
        # self.refine_block = FlotRefine()
        #PointNetFeaturePropogation(in_channel=128, mlp=[3])
        self.refine = DiffusionSceneFlowRefine(num_neighbors, in_channel=cross_mlp2[-1]+feat_ch, latent_channel=latent_ch, mlp=flow_channels, channels=flow_channels)
    def forward(self, p, feature1 ,feature2,interval):
        num_iters = 8
        # feature extraction
        [xyz1, xyz2] = p  #torch.Size([32, 256, 3])
        
        fmap1, graph = self.feature_extractor(p[0])
        fmap2, _ = self.feature_extractor(p[1])  
        
        
        start_time = time.time()
        # correlation matrix
        self.corr_block.init_module(fmap1, fmap2, xyz2)
        fct1, graph_context = self.context_extractor(p[0])
        end = time.time() - start_time
        print(end)
        
        
        net, inp = torch.split(fct1, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)


        coords1, coords2 = xyz1, xyz1
        flow_predictions = []

        for itr in range(num_iters):
            coords2 = coords2.detach()
            corr = self.corr_block(coords=coords2)
            flow = coords2 - coords1
            net, delta_flow = self.update_block(net, inp, corr, flow, graph_context)
            
            coords2 = coords2 + delta_flow
            flow_predictions.append(coords2 - coords1)
        
        flow = self.refine(flow_predictions[-1],graph)
        
        
        return flow_predictions


class DiffusionSceneFlowRefine(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, mlp, mlp2=None, bn = False, use_leaky = True, \
                 return_inter=False, radius=None, use_relu=False, channels = [64,64], clamp = [-200,200],scale_dif = 1.0):
        super(DiffusionSceneFlowRefine,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.use_relu = use_relu
        self.fc = nn.Conv1d(channels[-1],4,1)
        self.clamp = clamp

        #last_channel = in_channel + 3
        last_channel = in_channel + 3 + 64 + 3 + 1

        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

        # build diffusion
        timesteps = 1000
        sampling_timesteps = 1
        self.timesteps = timesteps
        # define beta schedule
        betas = cosine_beta_schedule(timesteps=timesteps).float()
        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0.01
        self.scale = scale_dif
        # time embeddings
        time_dim = 64
        dim = 16
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', log_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)



        self.iters = 1
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.scale*torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def forward(self, xyz1, xyz2, points1, points2, flow, flow_pseudo_gt, certainty, uncertainty=0.5):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)
        
        # add
        batch_size = flow.shape[0]
        n = flow.shape[2]

        if self.training:
            flow_gt = flow_gt.permute(0,2,1)
            certainty = certainty.permute(0,2,1)

            gt_certainty_norm = torch.norm(flow_gt - flow, dim=-2)
            sf_norm = torch.norm(flow_gt, dim=-2)
            relative_err = gt_certainty_norm / (sf_norm + 1e-4)

            def z_score_normalize(data):
                mean = np.mean(data, axis=1)
                std_dev = np.std(data, axis=1)
                normalized_data = (data - mean) / std_dev
                return normalized_data

            #gt_certainty = z_score_normalize(gt_certainty_norm)
            gt_certainty = torch.where(torch.logical_or(gt_certainty_norm < uncertainty, relative_err < uncertainty), torch.ones_like(gt_certainty_norm), torch.zeros_like(gt_certainty_norm))


            gt_certainty = torch.unsqueeze(gt_certainty, dim=2)

            gt_delta_certainty = gt_certainty - certainty
            gt_delta_certainty = gt_delta_certainty.detach()
            gt_delta_flow = flow_gt - flow
            gt_delta_flow = torch.where(torch.isinf(gt_delta_flow), torch.zeros_like(gt_delta_flow), gt_delta_flow)
            gt_delta_flow = gt_delta_flow.detach()

            t = torch.randint(0, self.timesteps, (batch_size,), device= flow.device).long()
            noise = (self.scale * torch.randn_like(gt_delta_flow)).float()
            noise_certainty = (self.scale * torch.randn_like(gt_delta_certainty)).float()

            delta_flow = self.q_sample(x_start=gt_delta_flow, t=t, noise=noise)
            flow_new = flow + delta_flow
            delta_certainty = self.q_sample(x_start=gt_delta_certainty, t=t, noise=noise_certainty)
            certainty_new = certainty + delta_certainty

            for i in range(self.iters):
                delta_flow = delta_flow.detach()
                flow_new = flow_new.detach()
                time = self.time_mlp(t)
                delta_certainty = delta_certainty.detach()
                certainty_new = certainty_new.detach()
                # #print(time.shape)
                time = time.unsqueeze(1).repeat(1, n, 1)
  
                if self.radius is None:
                    sqrdists = square_distance(xyz1, xyz2)
                    dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
                    neighbor_xyz = index_points_group(xyz2, knn_idx)
                    direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

                    grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
                    time = time.unsqueeze(-2).repeat(1,1,self.nsample,1)
                    delta_flow = delta_flow.permute(0,2,1)
                    delta_flow = delta_flow.unsqueeze(-2).repeat(1,1,self.nsample,1)
                    delta_certainty = delta_certainty.unsqueeze(-2).repeat(1,1,self.nsample,1)

                    new_points = torch.cat([grouped_points2, direction_xyz, delta_certainty, delta_flow, time], dim = -1) # B, N1, nsample, D1+D2+3

                    new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]

                else:
                    new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
                    new_points = new_points.permute(0, 1, 3, 2)

                point1_graph = points1

                # r
                r = new_points
                for i, conv in enumerate(self.mlp_r_convs):
                    r = conv(r)
                    if i == 0:
                        grouped_points1 = self.fuse_r(point1_graph)
                        r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                    if self.bn:
                        r = self.mlp_r_bns[i](r)
                    if i == len(self.mlp_r_convs) - 1:
                        r = self.sigmoid(r)
                    else:
                        r = self.relu(r)


                # z
                z = new_points
                for i, conv in enumerate(self.mlp_z_convs):
                    z = conv(z)
                    if i == 0:
                        grouped_points1 = self.fuse_z(point1_graph)
                        z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                    if self.bn:
                        z = self.mlp_z_bns[i](z)
                    if i == len(self.mlp_z_convs) - 1:
                        z = self.sigmoid(z)
                    else:
                        z = self.relu(z)

                    if i == len(self.mlp_z_convs) - 2:
                        z = torch.max(z, -2)[0].unsqueeze(-2)
                
                z = z.squeeze(-2)

                point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                point1_expand = r * point1_graph_expand
                point1_expand = self.fuse_r_o(point1_expand)

                h = new_points
                for i, conv in enumerate(self.mlp_h_convs):
                    h = conv(h)
                    if i == 0:
                        h = h + point1_expand
                    if self.bn:
                        h = self.mlp_h_bns[i](h)
                    if i == len(self.mlp_h_convs) - 1:
                        # 
                        if self.use_relu:
                            h = self.relu(h)
                        else:
                            h = self.tanh(h)
                    else:
                        h = self.relu(h)
                    if i == len(self.mlp_h_convs) - 2:
                        h = torch.max(h, -2)[0].unsqueeze(-2)

                h = h.squeeze(-2)

                new_points = (1 - z) * points1 + z * h

                if self.mlp2:
                    for _, conv in enumerate(self.mlp2):
                        new_points = conv(new_points)        

                new_points_delta = new_points - points1
                update = self.fc(new_points_delta)
                delta_flow, delta_certainty = update[:, :3, :].clamp(self.clamp[0], self.clamp[1]), update[:, 3:, :]
                certainty = certainty.permute(0,2,1)
                certainty_new = certainty + delta_certainty
                if flow is None:
                    flow = delta_flow
                else:
                    flow_new = delta_flow + flow
                
                loss_df = F.mse_loss(delta_flow, gt_delta_flow)
                gt_delta_certainty = gt_delta_certainty.permute(0,2,1)
                loss_dc = F.mse_loss(delta_certainty, gt_delta_certainty)
                loss = loss_df + loss_dc
            return new_points, flow_new, certainty_new, loss
        else:
            batch, device, total_timesteps, sampling_timesteps, eta = batch_size, flow.device, self.timesteps, self.sampling_timesteps, self.ddim_sampling_eta

            times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

            img = (self.scale * torch.randn_like(flow)).float()
            img_certainty = (self.scale * torch.randn_like(certainty)).float()

            #for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            for time, time_next in time_pairs:
                t = torch.full((batch,), time, device=device, dtype=torch.long)

                delta_flow = img
                flow_new = flow + delta_flow

                delta_certainty = img_certainty
                certainty_new = certainty + delta_certainty

                for i in range(self.iters):
                    delta_flow = delta_flow.detach()
                    flow_new = flow_new.detach()

                    delta_certainty = delta_certainty.detach()
                    certainty_new = certainty_new.detach()

                    time = self.time_mlp(t)
                    # #print(time.shape)
                    time = time.unsqueeze(1).repeat(1, n, 1)
    
                    if self.radius is None:
                        sqrdists = square_distance(xyz1, xyz2)
                        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
                        neighbor_xyz = index_points_group(xyz2, knn_idx)
                        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

                        grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
                        time = time.unsqueeze(-2).repeat(1,1,self.nsample,1)
                        delta_flow = delta_flow.permute(0,2,1)
                        delta_flow = delta_flow.unsqueeze(-2).repeat(1,1,self.nsample,1)
                        delta_certainty = delta_certainty.permute(0,2,1)
                        delta_certainty = delta_certainty.unsqueeze(-2).repeat(1,1,self.nsample,1)


                        #points1_1 = points1.permute(0, 2, 1)
                        #points1_1 = points1_1.unsqueeze(-2).repeat(1, 1, self.nsample, 1)

                        new_points = torch.cat([grouped_points2, direction_xyz, delta_certainty, delta_flow, time], dim = -1) # B, N1, nsample, D1+D2+3
                        new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]

                    else:
                        new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
                        new_points = new_points.permute(0, 1, 3, 2)

                    point1_graph = points1

                    # r
                    r = new_points
                    for i, conv in enumerate(self.mlp_r_convs):
                        r = conv(r)
                        if i == 0:
                            grouped_points1 = self.fuse_r(point1_graph)
                            r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                        if self.bn:
                            r = self.mlp_r_bns[i](r)
                        if i == len(self.mlp_r_convs) - 1:
                            r = self.sigmoid(r)
                        else:
                            r = self.relu(r)


                    # z
                    z = new_points
                    for i, conv in enumerate(self.mlp_z_convs):
                        z = conv(z)
                        if i == 0:
                            grouped_points1 = self.fuse_z(point1_graph)
                            z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                        if self.bn:
                            z = self.mlp_z_bns[i](z)
                        if i == len(self.mlp_z_convs) - 1:
                            z = self.sigmoid(z)
                            # #print('sigmoid', z.shape)
                        else:
                            z = self.relu(z)
                            # #print('relu', z.shape)

                        if i == len(self.mlp_z_convs) - 2:
                            z = torch.max(z, -2)[0].unsqueeze(-2)
                            # #print('max', z.shape)
                    
                    z = z.squeeze(-2)

                    point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                    point1_expand = r * point1_graph_expand
                    point1_expand = self.fuse_r_o(point1_expand)

                    h = new_points
                    for i, conv in enumerate(self.mlp_h_convs):
                        h = conv(h)
                        if i == 0:
                            h = h + point1_expand
                        if self.bn:
                            h = self.mlp_h_bns[i](h)
                        if i == len(self.mlp_h_convs) - 1:
                            # 
                            if self.use_relu:
                                h = self.relu(h)
                            else:
                                h = self.tanh(h)
                        else:
                            h = self.relu(h)
                        if i == len(self.mlp_h_convs) - 2:
                            h = torch.max(h, -2)[0].unsqueeze(-2)

                    h = h.squeeze(-2)

                    new_points = (1 - z) * points1 + z * h

                    if self.mlp2:
                        for _, conv in enumerate(self.mlp2):
                            new_points = conv(new_points)        
                    
                    new_points_delta = new_points - points1
                    #delta_flow = self.fc(new_points_delta).clamp(self.clamp[0], self.clamp[1]) 
                    update = self.fc(new_points_delta)
                    delta_flow, delta_certainty = update[:, :3, :].clamp(self.clamp[0], self.clamp[1]), update[:, 3:, :]
                    
                    certainty_new = certainty + delta_certainty

                    if flow is None:
                        flow_new = delta_flow
                    else:
                        flow_new = delta_flow + flow


                pred_noise = self.predict_noise_from_start(img, t, delta_flow)
                pred_noise_certainty = self.predict_noise_from_start(img_certainty, t, delta_certainty)

                if time_next < 0:
                    delta_flow = delta_flow
                    delta_certainty = delta_certainty
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = (self.scale * torch.randn_like(flow)).float()
                noise_certainty = (self.snr_scale * torch.randn_like(certainty)).float()

                img = delta_flow * alpha_next.sqrt() + \
                      c * pred_noise + \
                      sigma * noise
                img_certainty = delta_certainty * alpha_next.sqrt() + c * pred_noise_certainty + sigma * noise_certainty
            
            return new_points, flow_new, certainty_new
