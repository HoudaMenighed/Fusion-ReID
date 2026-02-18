#==================== Training Account : menmecho@gmail.com (V4)

import torch
import torch.nn as nn
from modeling.backbones.resnet import ResNet, Bottleneck
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.fusion_part.fusion import Heterogenous_Transmission_Module
import torch.nn.functional as F
from modeling.backbones.t2tvit import t2t_vit_t_24, t2t_vit_t_14
from modeling.backbones.saph_resnet import SAPH_MultiScale_CBAM
# from mmcv.ops import DeformConv2dPack
import math


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class build_resnet(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_resnet, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH_R
        self.mode = cfg.MODEL.TRANS_USE
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        self.in_planes = 2048
        self.pattern = cfg.MODEL.RES_MODE

        # 1. Standard ResNet Backbone
        if self.pattern == 1:
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # --- SAPH-Net INTEGRATION START ---
        # 2. Add the SAPH block (Multi-Scale + Parallel CBAM)
        # Ensure SAPH_MultiScale_CBAM is imported or defined in the same file
        self.saph_block = SAPH_MultiScale_CBAM(self.in_planes)
        # --- SAPH-Net INTEGRATION END ---

        self.gap = GeM()
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, cam_label=None, view_label=None, label=None):
        # 1. Extract raw features from ResNet (typically [B, 2048, 16, 8] or [B, 2048, 8, 4])
        mid_fea = self.base(x)

        # 2. Apply SAPH refinement (Multi-Scale + CBAM)
        # mid_fea_refined will be used for both global pooling and future part-masking
        mid_fea_refined = self.saph_block(mid_fea)

        # 3. Global Pooling on the REFINED features
        global_feat = self.gap(mid_fea_refined)
        global_feat = global_feat.view(global_feat.shape[0], -1)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            if self.mode == 0:
                return mid_fea_refined, cls_score, global_feat
            else:
                # Return the refined multi-scale features for the Transformer/Fusion branch
                return mid_fea_refined, cls_score, global_feat
        else:
            cls_score = None
            if self.neck_feat == 'after':
                if self.mode == 0:
                    return feat
                else:
                    return mid_fea_refined,cls_score, feat
            else:
                if self.mode == 0:
                    return global_feat
                else:
                    return mid_fea_refined,cls_score, global_feat
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        self.mode = cfg.MODEL.RES_USE
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        if 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 512
        if 'edge' in cfg.MODEL.TRANSFORMER_TYPE or cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224':
            self.in_planes = 384
        if '14' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 384
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))


        camera_num = 0

        view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        num_classes=num_classes,
                                                        camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=0, view_label=None):
        mid_fea = self.base(x, cam_label=cam_label, view_label=view_label)
        global_feat = mid_fea[:, 0]
        mid_fea_f = mid_fea[:, 1:, :]
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            if self.mode == 0:
                return mid_fea_f, cls_score, global_feat
            else:
                return mid_fea_f, cls_score, global_feat  # global feature for triplet loss
        else:
            cls_score = None
            if self.neck_feat == 'after':
                if self.mode == 0:
                    return feat
                else:
                    return mid_fea_f, feat
            else:
                if self.mode == 0:
                    return global_feat
                else:
                    return mid_fea_f, cls_score, global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class LocalRefinementUnits(nn.Module):
    def __init__(self, dim, out_dim=768, kernel=1, choice=True):
        super().__init__()
        self.LRU = choice
        self.channels = dim
        self.out_dim = out_dim
        self.dwconv = nn.Conv2d(self.channels, self.channels, kernel, 1, padding=0, groups=self.channels)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.ptconv = nn.Conv2d(self.channels, self.out_dim, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.out_dim)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act = nn.ReLU()

    def forward(self, x):
        if self.LRU:
            x = self.act1(self.bn1(self.dwconv(x)))
            x = self.act2(self.bn2(self.ptconv(x)))
        else:
            x = self.act2(self.bn2(self.ptconv(x)))
        return x


class FusionReID(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(FusionReID, self).__init__()

        # Backbone branches
        self.resnet = build_resnet(num_classes, cfg)
        self.transformer = build_transformer(
            num_classes, cfg, camera_num, view_num, factory
        )

        self.num_classes = num_classes
        self.mix_dim = cfg.MODEL.MIX_DIM

        # Projection Layers (LRU)
        self.res_LRU = LocalRefinementUnits(dim=2048, out_dim=self.mix_dim)

        dim_l = 384 if '14' in cfg.MODEL.TRANSFORMER_TYPE else 768
        self.former_LRU = LocalRefinementUnits(dim=dim_l, out_dim=self.mix_dim)

        # Asymmetric Cross Attention
        self.CAF = nn.MultiheadAttention(
            embed_dim=self.mix_dim,
            num_heads=8,
            batch_first=True
        )

        # ---- NEW: Dynamic Gating Mechanism ----
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.mix_dim * 2, self.mix_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mix_dim, self.mix_dim),
            nn.Sigmoid()
        )

        # Pooling for CNN global descriptor (for gating)
        self.gap_r = GeM()

        # Classifier
        self.bottleneck_3 = nn.BatchNorm1d(self.mix_dim)
        self.classifier_3 = nn.Linear(self.mix_dim, self.num_classes, bias=False)

    def forward(self, x, label=None, cam_label=0, view_label=None):
        B = x.shape[0]

        # ---- STEP 1: Feature Extraction ----
        mid_fea_r, cls_score_r, global_feat_r = self.resnet(x)
        mid_fea_f, cls_score_f, global_feat_f = self.transformer(
            x, cam_label=cam_label, view_label=view_label
        )

        # ---- STEP 2: Feature Alignment ----

        # CNN spatial tokens
        mid_fea_r = self.res_LRU(mid_fea_r)              # [B, mix_dim, 8, 4]
        cnn_spatial_tokens = mid_fea_r.flatten(2).permute(0, 2, 1)  # [B, 32, mix_dim]

        # Transformer CLS token projection
        query_f = self.former_LRU(
            global_feat_f.unsqueeze(-1).unsqueeze(-1)
        ).view(B, 1, self.mix_dim)  # [B, 1, mix_dim]

        transformer_feat = query_f.squeeze(1)  # [B, mix_dim]

        # ---- STEP 3: Asymmetric Cross-Attention ----
        cross_feat, _ = self.CAF(
            query_f,
            cnn_spatial_tokens,
            cnn_spatial_tokens
        )
        cross_feat = cross_feat.squeeze(1)  # [B, mix_dim]

        # ---- STEP 4: Dynamic Gating ----

        # CNN global descriptor (for gate decision)
        cnn_global = self.gap_r(mid_fea_r).view(B, self.mix_dim)

        # Concatenate Transformer + CNN global
        gate_input = torch.cat([transformer_feat, cnn_global], dim=1)

        alpha = self.gate_mlp(gate_input)  # [B, mix_dim]

        # ---- STEP 5: Gated Residual Fusion ----
        fused_feat = transformer_feat + alpha * cross_feat

        # ---- STEP 6: Classification ----
        feat_3 = self.bottleneck_3(fused_feat)
        cls_score_3 = self.classifier_3(feat_3)

        if self.training:
            return (
                cls_score_r, global_feat_r,
                cls_score_f, global_feat_f,
                cls_score_3, fused_feat
            )
        else:
            return F.normalize(fused_feat, p=2, dim=1)



__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_24': t2t_vit_t_24,
    't2t_vit_t_14': t2t_vit_t_14
}


def make_model(cfg, num_class, camera_num, view_num=0, ):
    if cfg.MODEL.RES_USE and not cfg.MODEL.TRANS_USE:
        model = build_resnet(num_class, cfg)
        print('===========Building ResNet Only===========')
        return model
    elif cfg.MODEL.TRANS_USE and not cfg.MODEL.RES_USE:
        model = build_transformer(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building Transformer Only===========')
        return model
    elif cfg.MODEL.TRANS_USE and cfg.MODEL.RES_USE:
        model = FusionReID(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building FusionReID===========')
        return model
    else:
        print("===========Fail to build model,Please check cfg.MODEL.RES_USE and cfg.MODEL.TRANS_USE===========")
        return None
