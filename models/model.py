import torch.nn as nn
import torch
import numpy as np
import os
import torch.nn.functional as F
from utils.model_utils import *
from .jgwtf import *
from .raflow import *
from .cmflow import *
from .pointpwcnet import *
from .flot.flot import *
from .pvraft.RAFTSceneFlow import RSF
from .flownet3d import *
from .flowstep3d import *
from .slim import *
from .cmflow_t import *
from .cmflow_b import *


def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1 and classname.find('Conv1d_p') == -1:
        nn.init.kaiming_normal_(m.weight.data)
        
def init_model(args):
    
    if args.model in ['jgwtf','pointpwcnet','raflow','gl', 'flownet3d', 'pointpwcnet_full', 'flot', 'flowstep3d_full', \
                      'pvraft', 'flowstep3d','slim','cmflow_o', 'cmflow_ol', \
                      'cmflow_oc', 'cmflow_c', 'cmflow_olc', 'cmflow_lc', \
                      'cmflowt_o', 'cmflowt_ol', 'cmflowt_olc', 'cmflow_l', 'cmflow_o', 'cmflow_t', 'cmflow_b']:
        if args.model =='jgwtf':
            net = JGWTF(args).cuda()
        if args.model == 'pvraft':
            net = RSF(args).cuda()
        if args.model in ['gl', 'flownet3d']:
            net = FlowNet3D(args).cuda()
        if args.model == 'flot':
            net = FLOT().cuda()
        if args.model == 'pointpwcnet_full':
            net = PointPWCNet(args).cuda()
        if args.model == 'flowstep3d_full':
            net = FlowStep3D(args).cuda()
        if args.model =='pointpwcnet':
            net = PointPWCNet(args).cuda()
        if args.model in ['raflow']:
            net = RaFlow(args).cuda()
        if args.model in ['cmflow_b']:
            net = CMFlow_B(args).cuda()
        if args.model in ['cmflow_l', 'cmflow_olc', 'cmflow_o', 'cmflow_ol', 'cmflow_oc']:
            net = CMFlow(args).cuda()
        if args.model in ['cmflow_c','cmflow_lc']:
            net = RaFlow(args).cuda()
        if args.model == 'flowstep3d':
            net = FlowStep3D(args).cuda()
        if args.model == 'slim':
            net = SLIM(args).cuda()
        if args.model in ['cmflow_t', 'cmflowt_o', 'cmflowt_ol', 'cmflowt_olc']:
            net = CMFlow_T(args).cuda()
        #if not args.model in ['cmflow_olc', 'raflow','slim', 'cmflow_o', 'cmflow_ol', 'arfnet_l','arfnet_s','cmflow_oc', 'cmflow_c']:
        #    net.apply(weights_init)  
            
        if args.eval or args.load_checkpoint:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
            print("Successfully load model parameters!")
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Experiment with", torch.cuda.device_count(), "GPUs!")
            
        return net
    
    elif args.model =='icp' or args.model =='gl_wo' or args.model == 'arfnet_o':
        print("Evaluate non-parametric method {}".format(args.model))
        
    else:
        raise Exception('Not implemented')
        
        
