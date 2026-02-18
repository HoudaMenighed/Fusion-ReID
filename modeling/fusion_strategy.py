#==================== Training Account : nhou777@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionReID(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(FusionReID, self).__init__()

        # Backbone branches
        self.resnet = build_resnet(num_classes, cfg)
        self.transformer = build_transformer(num_classes, cfg, camera_num, view_num, factory)

        self.num_classes = num_classes
        self.mix_dim = cfg.MODEL.MIX_DIM

        # Projection layers (alignment to mix_dim)
        self.res_LRU = LocalRefinementUnits(dim=2048, out_dim=self.mix_dim)

        if '14' in cfg.MODEL.TRANSFORMER_TYPE:
            dim_l = 384
        else:
            dim_l = 768

        self.former_LRU = LocalRefinementUnits(dim=dim_l, out_dim=self.mix_dim)

        # Spatial Cross-Attention (Transformer → CNN)
        self.cross_attn_tc = nn.MultiheadAttention(
            embed_dim=self.mix_dim,
            num_heads=8,
            batch_first=True
        )

        # Optional reverse attention (CNN → Transformer)
        self.cross_attn_ct = nn.MultiheadAttention(
            embed_dim=self.mix_dim,
            num_heads=8,
            batch_first=True
        )

        # Bottleneck + classifier for fused feature
        self.bottleneck = nn.BatchNorm1d(self.mix_dim * 2)
        self.bottleneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(self.mix_dim * 2, self.num_classes, bias=False)

    def forward(self, x, label=None, cam_label=0, view_label=None):

        B = x.shape[0]

        # -------------------------------------------------
        # STEP 1 — Extract Features from Both Branches
        # -------------------------------------------------

        # CNN branch
        mid_fea_r, cls_score_r, global_feat_r = self.resnet(x)
        # mid_fea_r: [B, 2048, H, W]

        # Transformer branch
        mid_fea_f, cls_score_f, global_feat_f = self.transformer(
            x,
            cam_label=cam_label,
            view_label=view_label
        )
        # mid_fea_f: [B, Tokens, dim_l]

        # -------------------------------------------------
        # STEP 2 — Convert to Spatial Tokens
        # -------------------------------------------------

        # ---- CNN tokens ----
        mid_fea_r = self.res_LRU(mid_fea_r)  # [B, mix_dim, H, W]
        cnn_tokens = mid_fea_r.flatten(2).permute(0, 2, 1)
        # [B, N_cnn, mix_dim]

        # ---- Transformer tokens (remove CLS if exists) ----
        if mid_fea_f.shape[1] > 1:
            trans_tokens = mid_fea_f[:, 1:, :]  # remove CLS token
        else:
            trans_tokens = mid_fea_f

        # reshape for projection
        trans_tokens = trans_tokens.permute(0, 2, 1).unsqueeze(-1)
        trans_tokens = self.former_LRU(trans_tokens)
        trans_tokens = trans_tokens.squeeze(-1).permute(0, 2, 1)
        # [B, N_trans, mix_dim]

        # -------------------------------------------------
        # STEP 3 — Full Spatial Cross-Attention Fusion
        # -------------------------------------------------

        # Transformer attends to CNN
        fused_t, _ = self.cross_attn_tc(
            query=trans_tokens,
            key=cnn_tokens,
            value=cnn_tokens
        )
        # [B, N_trans, mix_dim]

        # CNN attends to Transformer
        fused_c, _ = self.cross_attn_ct(
            query=cnn_tokens,
            key=trans_tokens,
            value=trans_tokens
        )
        # [B, N_cnn, mix_dim]

        # -------------------------------------------------
        # STEP 4 — Global Pooling
        # -------------------------------------------------

        fused_t_global = fused_t.mean(dim=1)
        fused_c_global = fused_c.mean(dim=1)

        fused_feat = torch.cat([fused_t_global, fused_c_global], dim=1)
        # [B, mix_dim * 2]

        # -------------------------------------------------
        # STEP 5 — Classification
        # -------------------------------------------------

        feat_bn = self.bottleneck(fused_feat)
        cls_score_fused = self.classifier(feat_bn)

        # -------------------------------------------------
        # TRAIN / TEST Behavior
        # -------------------------------------------------

        if self.training:
            return (
                cls_score_r, global_feat_r,
                cls_score_f, global_feat_f,
                cls_score_fused, fused_feat
            )
        else:
            return F.normalize(fused_feat, p=2, dim=1)
