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
                return cls_score, global_feat
            else:
                # Return the refined multi-scale features for the Transformer/Fusion branch
                return mid_fea_refined, cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                if self.mode == 0:
                    return feat
                else:
                    return mid_fea_refined, feat
            else:
                if self.mode == 0:
                    return global_feat
                else:
                    return mid_fea_refined, global_feat