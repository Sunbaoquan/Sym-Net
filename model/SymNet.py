import math
import torch
from torch import nn
import torch.nn.functional as F
from model.correlation import Correlation
from model.future import extract_feat_res, extract_feat_vgg
from functools import reduce
from operator import add
from model.PSPNet import OneModel as PSPNet
from util.util import load_obj


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (feat_h, feat_w)) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class TDC(nn.Module):
    """
    Top-down hyper-correlation module(TDC)
    """

    def __init__(self, in_channels, hidden_dim=16, out_channels=48):
        super(TDC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels[-1], out_channels=hidden_dim, kernel_size=1)  # l4
        self.conv2 = nn.Conv2d(in_channels=in_channels[-2], out_channels=hidden_dim, kernel_size=1)  # l3
        self.conv3 = nn.Conv2d(in_channels=in_channels[-3], out_channels=hidden_dim, kernel_size=1)  # l2
        self.merge1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.merge2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, padding=1)
        )

    def forward(self, hyper_2, hyper_3, hyper_4):
        out = self.merge1(self.conv2(hyper_3) + self.conv1(hyper_4))
        out = self.merge2(out + self.conv3(hyper_2))
        return out


class APA(nn.Module):
    """
    visual-text alignment prototype aggregation module(APA)
    """

    def __init__(self, embed_dim=556, hidden_dim=256, out_dim=256, score_function='bi_linear', dropout=0.1):
        """
            embed_dim: 300/1024 + 256
            hidden_dim: 256
            out_dim: 256
            score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
            Q: text+pro=[B, 556]  KV: [B 256 h w]  ==>  b 256
        """
        super(APA, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.score_function = score_function
        self.w_q = nn.Linear(embed_dim, hidden_dim, bias=False)  # 556 -> 256
        self.w_k = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1,
                             bias=False)  # 256 -> 256
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, text, prototype):
        # k: feat    (b, 256, h, w)
        # text: text (b, 1, 300)
        # prototype: (b, 256, 1, 1)
        assert len(k.shape) == 4
        bs, c, h, w = k.shape

        if len(text.shape) == 2:
            text = torch.unsqueeze(text, dim=1)
        if len(prototype.shape) == 4:
            prototype = prototype.view(bs, c).unsqueeze(1)

        q = torch.cat([text, prototype], -1)  # b 1 556

        q_len = q.shape[1]  # 1
        k_len = h * w

        kx = self.w_k(k).view(bs, c, -1)  # b hidden_dim hw
        kx = kx.permute(0, 2, 1).contiguous()  # (b, hw, hidden_dim)

        qx = self.w_q(q).view(bs, q_len, self.hidden_dim)  # (b 1 hidden_dim)
        # qx = qx.permute(0, 2, 1)  #(b, 1, hidden_dim)

        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)  # (b, hidden_dim, hw)
            score = torch.bmm(qx, kt)  # (n_head*b, 1, hw)

        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)  # (b, 1, hidden_dim)
            kt = kx.permute(0, 2, 1)  # (b, hidden_dim, hw)
            score = torch.bmm(qw, kt)  # (b, 1, hw)
        else:
            raise RuntimeError('invalid score_function')

        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (b, 1, hidden_dim)
        output = self.proj(output)  # (b, 1, out_dim)
        output = self.dropout(output)

        return output + prototype


def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)  # C*N
    fea_T = fea.permute(0, 2, 1)  # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)  # C*C
    return gram


def SPM(weight, query_feat_4, final_supp_list, mask_list, scale):
    """
    self-activation based prior mask generation module(SPM)
    """
    cosine_eps = 1e-7
    corr_query_mask_list = []
    for i, tmp_supp_feat in enumerate(final_supp_list):  # shot x high-level features
        bsize, ch_sz, _, _ = query_feat_4.size()
        assert tmp_supp_feat.size(2) == query_feat_4.size(2) == scale[0]

        # ------------------------ scale 60 ---------------------
        kernels = [(5, 5), (7, 1), (1, 7)]
        tmp_mask = F.interpolate(mask_list[i], size=(scale[0], scale[0]), mode='bilinear', align_corners=True)
        s = tmp_supp_feat * tmp_mask
        query_high = query_feat_4  # 60 x 60
        q = query_high

        region_mask_list = []
        for kernel in kernels:
            tmp_s = nn.AvgPool2d(kernel_size=kernel, stride=1, padding=(kernel[0] // 2, kernel[1] // 2))(s).view(bsize,
                                                                                                                 ch_sz,
                                                                                                                 -1)
            s_norm = torch.norm(tmp_s, 2, 1, True)
            tmp_q = nn.AvgPool2d(kernel_size=kernel, stride=1, padding=(kernel[0] // 2, kernel[1] // 2))(q).view(bsize,
                                                                                                                 ch_sz,
                                                                                                                 -1)
            tmp_q = tmp_q.contiguous().permute(0, 2, 1)
            q_norm = torch.norm(tmp_q, 2, 2, True)
            similarity = torch.matmul(tmp_q, tmp_s) / (
                    torch.matmul(q_norm, s_norm) + cosine_eps) * q_norm  # B HW(q) HW(s)
            similarity = similarity.mean(2).view(bsize, scale[0] * scale[0])
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                    similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            similarity = similarity.view(bsize, 1, scale[0], scale[0])
            region_mask_list.append(similarity)
        corr_query = torch.cat(region_mask_list, 1).mean(1).unsqueeze(1)
        corr_query_mask_list.append(corr_query)

    corr_query_mask_1 = torch.cat(corr_query_mask_list, 1)  # [b, shot, 1, 1]
    corr_query_mask_1 = (weight * corr_query_mask_1).sum(1, True)

    corr_query_mask_2 = F.interpolate(corr_query_mask_1, size=(scale[1], scale[1]), mode='bilinear', align_corners=True)
    corr_query_mask_3 = F.interpolate(corr_query_mask_1, size=(scale[2], scale[2]), mode='bilinear', align_corners=True)
    corr_query_mask_4 = F.interpolate(corr_query_mask_1, size=(scale[3], scale[3]), mode='bilinear', align_corners=True)
    return [corr_query_mask_1, corr_query_mask_2, corr_query_mask_3, corr_query_mask_4]


def mask_feature(features, support_mask):
    bs = features[0].shape[0]
    for idx, feature in enumerate(features):
        feat = []
        if support_mask.shape[-1] != feature.shape[-1]:
            tmp_mask = F.interpolate(support_mask, feature.size()[2:], mode='bilinear', align_corners=True)
        for i in range(bs):
            featI = feature[i].flatten(start_dim=1)  # c, hw
            maskI = tmp_mask[i].flatten(start_dim=1).squeeze()  # hw
            realSupI = featI[:, (maskI == 1)]
            if (maskI == 1).sum() == 0:
                realSupI = torch.zeros(featI.shape[0], 1, device='cuda')
            feat.append(realSupI)  # [ (c, N) x B]
        features[idx] = feat  # [[(c,N) x B] x nfeatures]
    return features


class SymNet(nn.Module):
    def __init__(self, args, cls_type=None):
        super(SymNet, self).__init__()
        assert args.classes > 1
        assert args.shot in [1, 5, 10, 20]
        assert args.data_set in ['pascal', 'coco']
        assert args.nlp in ['word2vec', 'glove', 'clip']

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.args = args
        self.zoom_factor = args.zoom_factor
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.shot = args.shot
        self.ppm_scales = args.scales
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.base_classes = 15 if self.dataset == 'pascal' else 60

        if args.nlp == 'word2vec':
            self.text_dim = 300
        elif args.nlp == 'glove':
            self.text_dim = 300
        elif args.nlp == 'clip':
            self.text_dim = 512
        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        PSPNet_ = PSPNet(args)
        backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)
        weight_path = './model/initmodel/PSPNet/{}/split{}/{}/best.pth'.format(self.dataset, args.split, backbone_str)
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:  # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4

        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)

        self.low_fea_id = args.low_fea[-1]

        if backbone_str == 'vgg':
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
            self.nsimlairy = [1, 3, 3]  # 7
        elif backbone_str == 'resnet50':
            self.feat_ids = list(range(3, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
            self.nsimlairy = [3, 6, 4]  # 13
        elif backbone_str == 'resnet101':
            self.feat_ids = list(range(3, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
            self.nsimlairy = [3, 23, 4]  # 30
        else:
            raise Exception('Unavailable backbone: %s' % backbone_str)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, args.classes, kernel_size=1)
        )
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.pyramid_bins = args.scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        mask_add_num = 1 + 48
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, args.classes, kernel_size=1)
            ))
        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins) - 1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(2*reduce_dim, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))
        self.alpha_conv = nn.ModuleList(self.alpha_conv)

        self.hyper_final = TDC(in_channels=nbottlenecks[1:], out_channels=48)

        # Gram and Meta
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))

        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = 2  # args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))
        self.sigmoid = nn.Sigmoid()

        # -----------------text prompt-------------------
        self.CrossAttention = APA(embed_dim=self.text_dim + reduce_dim)

    # que_img, sup_img, sup_mask, que_mask(meta), que_mask(base), cat_idx(meta), class_chosen
    def forward(self, x, s_x, s_y, y, y_b, cat_idx, class_chosen):
        """
            x: [B, 3, 473, 473]
            y: [B, 473, 473]
            s_x: [B, shot, 3, 473, 473]
            s_y: [B, shot, 473, 473]
        """
        x_size = x.size()
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # embeddings [21, 300]  coco: [80, 300]
        device = torch.device('cuda')
        embeddings = load_obj('./embeddings/{}_{}'.format(self.args.nlp, self.dataset)).to(device)
        embeddings = torch.stack([embeddings] * bs, dim=0)
        row = list(range(embeddings.shape[0]))
        embeddings = embeddings[row, class_chosen]


        with torch.no_grad():
            query_feats, query_backbone_layers = self.extract_feats(x,
                                                                    [self.layer0, self.layer1, self.layer2, self.layer3,
                                                                     self.layer4],
                                                                    self.feat_ids,
                                                                    self.bottleneck_ids,
                                                                    self.lids)
        if self.vgg:
            query_feat = F.interpolate(query_backbone_layers[2],
                                       size=(query_backbone_layers[3].size(2), query_backbone_layers[3].size(3)),
                                       mode='bilinear', align_corners=True)
            query_feat = torch.cat([query_backbone_layers[3], query_feat], 1)
        else:
            query_feat = torch.cat([query_backbone_layers[3], query_backbone_layers[2]], 1)

        query_feat = self.down_query(query_feat)

        supp_pro_list = []  # support prototypes
        final_supp_list = []  # high-level features
        mid_supp_list = []
        mask_list = []  # [B 1 H W] x shot
        supp_feat_list = []  # low-level features
        corrs = []  # [corr4, corr3, corr2] x shot
        supp_text = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)  # [B 1 H W]: {0, 1}
            mask_list.append(mask)
            with torch.no_grad():
                support_feats, support_backbone_layers = self.extract_feats(s_x[:, i, :, :, :],
                                                                            [self.layer0, self.layer1, self.layer2,
                                                                             self.layer3, self.layer4],
                                                                            self.feat_ids, self.bottleneck_ids,
                                                                            self.lids)
                final_supp_list.append(support_backbone_layers[4])
                if self.vgg:
                    supp_feat = F.interpolate(support_backbone_layers[2], size=(
                        support_backbone_layers[3].size(2), support_backbone_layers[3].size(3)),
                                              mode='bilinear', align_corners=True)
                    supp_feat = torch.cat([support_backbone_layers[3], supp_feat], 1)
                else:
                    supp_feat = torch.cat([support_backbone_layers[3], support_backbone_layers[2]], 1)

            mask_down = F.interpolate(mask,
                                      size=(support_backbone_layers[3].size(2), support_backbone_layers[3].size(3)),
                                      mode='bilinear', align_corners=True)  # {0, 1}
            supp_feat = self.down_supp(supp_feat)  # sup mid-level
            mid_supp_list.append(supp_feat)

            supp_pro = Weighted_GAP(supp_feat, mask_down)  # prototype
            supp_pro_list.append(supp_pro)

            # text-embedding  # [b c h w]   [b 300+256] -> [b 1 256]
            text = self.CrossAttention(k=supp_feat * mask_down, text=embeddings, prototype=supp_pro)
            supp_text.append(text)

            supp_feat_list.append(support_backbone_layers[2])  # sup low-level (layer-2)


            support_feats = mask_feature(support_feats, mask)
            corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)
            corrs.append(corr)


        corrs_shot = [corrs[0][i] for i in range(len(self.nsimlairy))]
        for ly in range(len(self.nsimlairy)):
            for s in range(1, self.shot):
                corrs_shot[ly] += (corrs[s][ly])
        hyper_4 = corrs_shot[0] / self.shot
        hyper_3 = corrs_shot[1] / self.shot
        if self.vgg:
            hyper_2 = F.interpolate(corrs_shot[2], size=(corrs_shot[1].size(2), corrs_shot[1].size(3)), mode='bilinear',
                                    align_corners=True)
            hyper_2 = hyper_2 / self.shot
        else:
            hyper_2 = corrs_shot[2] / self.shot

        hyper_final = self.hyper_final(hyper_2, hyper_3, hyper_4)

        # --------------------------K-Shot Reweighting--------------
        que_gram = get_gram_matrix(query_backbone_layers[2])  # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            weight = weight.gather(1, idx2)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

        corr_query_mask_list = SPM(weight_soft, query_backbone_layers[4], final_supp_list, mask_list,
                                   self.pyramid_bins)

        supp_text = torch.cat(supp_text, 1)  # [b, shot, 256]
        supp_text = (weight_soft.squeeze(-1) * supp_text).sum(1, True)  # [b, 1, 256]

        query_fg_pro, query_h_pro, query_bg_pro = self.feat_aggregation(query_feat,
                                                                        corr_query_mask_list[0])

        query_text = self.CrossAttention(k=query_feat * corr_query_mask_list[0],
                                         text=embeddings,
                                         prototype=query_fg_pro).view(bs, query_feat.size(1))
        supp_text = supp_text.view(bs, query_feat.size(1))  # b, 256

        if self.training:
            triple_loss = self.get_triple_loss(query_text, supp_text, query_h_pro, query_bg_pro, mask_list,
                                               mid_supp_list)
        else:
            triple_loss = torch.tensor([0.0]).to(device)

        general_prototype = 0.5 * query_text.unsqueeze(-1).unsqueeze(-1) + 0.5 * supp_text.unsqueeze(-1).unsqueeze(-1)

        out_list = []
        pyramid_feat_list = []
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)  # init_merge / query_feat√
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)  # init_merge / query_feat√
            general_prototype_bin = general_prototype.expand(-1, -1, bin, bin)
            hyper_final_bin = F.interpolate(hyper_final, size=(bin, bin), mode='bilinear', align_corners=True)
            corr_mask_bin = corr_query_mask_list[idx]
            merge_feat_bin = torch.cat([query_feat_bin, general_prototype_bin, hyper_final_bin, corr_mask_bin],
                                       1)  # 256+256+64+1
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        # Base and Meta
        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        meta_out = self.cls(query_feat)

        base_out = self.learner_base(query_backbone_layers[4])

        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # Classifier Ensemble
        meta_map_bg = meta_out_soft[:, 0:1, :, :]  # [bs, 1, 60, 60]
        meta_map_fg = meta_out_soft[:, 1:, :, :]  # [bs, 1, 60, 60]
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes + 1, device='cuda')
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array != 0) & (c_id_array != c_id)
                base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
            base_map = torch.cat(base_map_list, 0)
            # <alternative implementation>
            # gather_id = (cat_idx[0]+1).reshape(bs,1,1,1).expand_as(base_out_soft[:,0:1,:,:]).cuda()
            # fg_map = base_out_soft.gather(1,gather_id)
            # base_map = base_out_soft[:,1:,:,:].sum(1,True) - fg_map
        else:
            base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)  # [B, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        #   Output Part
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:

            main_loss = self.criterion(final_out, y.long())  # main loss
            aux_loss1 = self.criterion(meta_out, y.long())  # aux loss 1
            aux_loss2 = self.criterion(base_out, y_b.long())  # aux loss 2

            aux_loss4 = torch.zeros_like(main_loss, device='cuda')
            for idx_k in range(len(out_list)):
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss4 = aux_loss4 + self.criterion(inner_out, y.long())
            aux_loss4 = aux_loss4 / len(out_list)
            return final_out.max(1)[1], main_loss, aux_loss1, aux_loss2, aux_loss4, triple_loss
        else:
            return final_out, meta_out, base_out

    def feat_aggregation(self, feat_q, pm_q):
        """
            feat_q: [b, c, 60, 60]
            out: [B, 1, 60, 60]
        """
        bs, c, h, w = feat_q.size()
        pm_q = pm_q.view(bs, -1)

        fg_pros = []
        fg_h_pros = []
        bg_pros = []
        for i in range(bs):
            fg_thres = 0.7
            bg_thres = 0.4
            cur_feat = feat_q[i].view(c, -1)  # C, HW

            # aggregating foreground region
            if (pm_q[i] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pm_q[i] > fg_thres)]  # C,N1
            else:
                fg_feat = cur_feat[:, torch.topk(pm_q[i], 12).indices]
            fg_pro = fg_feat.mean(-1).unsqueeze(-1).unsqueeze(-1)  # C,1,1

            # aggregating ambiguous region
            if ((pm_q[i] > 0.4) & (pm_q[i] < 0.55)).sum() > 0:
                fg_h_feat = cur_feat[:, (pm_q[i] > 0.4) & (pm_q[i] < 0.55)]  # C,N2
            else:
                _, idx = pm_q[i].sort()
                fg_h_feat = cur_feat[:, idx[(h * w // 2) - 6:(h * w // 2) + 6]]
            fg_h_pro = fg_h_feat.mean(-1).unsqueeze(-1).unsqueeze(-1)

            # aggregating background region
            if (pm_q[i] < bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pm_q[i] < bg_thres)]  # C,N3
            else:
                bg_feat = cur_feat[:, torch.topk((1 - pm_q[i]), 12).indices]
            bg_pro = bg_feat.mean(-1).unsqueeze(-1).unsqueeze(-1)

            fg_pros.append(fg_pro.unsqueeze(0))  # 1,C,1,1
            fg_h_pros.append(fg_h_pro.unsqueeze(0))
            bg_pros.append(bg_pro.unsqueeze(0))

        fg_pro = torch.cat(fg_pros, 0)  # B,C,1,1
        fg_h_pro = torch.cat(fg_h_pros, 0)
        bg_pro = torch.cat(bg_pros, 0)
        return fg_pro, fg_h_pro, bg_pro

    def get_optim(self, model, args):
        if args.shot > 1:
            optimizer = torch.optim.SGD(
                [
                    {'params': model.hyper_final.parameters()},
                    {'params': model.down_query.parameters()},
                    {'params': model.down_supp.parameters()},
                    {'params': model.alpha_conv.parameters()},
                    {'params': model.init_merge.parameters()},
                    {'params': model.beta_conv.parameters()},
                    {'params': model.inner_cls.parameters()},
                    {'params': model.res1.parameters()},
                    {'params': model.res2.parameters()},
                    {'params': model.cls.parameters()},
                    {'params': model.gram_merge.parameters()},
                    {'params': model.cls_merge.parameters()},
                    {'params': model.kshot_rw.parameters()},
                    {'params': model.CrossAttention.parameters()},
                ],
                lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                [
                    {'params': model.hyper_final.parameters()},
                    {'params': model.down_query.parameters()},
                    {'params': model.down_supp.parameters()},
                    {'params': model.alpha_conv.parameters()},
                    {'params': model.init_merge.parameters()},
                    {'params': model.beta_conv.parameters()},
                    {'params': model.inner_cls.parameters()},
                    {'params': model.res1.parameters()},
                    {'params': model.res2.parameters()},
                    {'params': model.cls.parameters()},
                    {'params': model.gram_merge.parameters()},
                    {'params': model.cls_merge.parameters()},
                    {'params': model.CrossAttention.parameters()},
                ],
                lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.learner_base.parameters():
            param.requires_grad = False

    def get_triple_loss(self, q_x, s_x, query_hard_fg_pro, query_bg_pro, mask_list, mid_supp_list):

        shot = len(mask_list)
        assert len(s_x.shape) == 2 and len(q_x.shape) == 2
        b, c, h, w = mid_supp_list[0].shape
        query_hard_fg_pro = query_hard_fg_pro.view(b, c)
        query_bg_pro = query_bg_pro.view(b, c)

        q_triplet_loss = torch.tensor([0.0], device='cuda')
        s_triplet_loss = torch.tensor([0.0], device='cuda')

        for i in range(shot):
            tmp_mask = F.interpolate(mask_list[i].float(), size=(h, w), mode="nearest").long().view(b, -1)  # [b, h*w]
            tmp_s_feat = mid_supp_list[i].view(b, c, -1).permute(0, 2, 1).contiguous()  # [b, hw, c]
            shot_loss = torch.tensor([0.0], device='cuda')
            count = b
            for j in range(b):
                mask_j = tmp_mask[j]  # [hw]
                negative_list_j = tmp_s_feat[j][mask_j == 0]  # [n1, c]
                positive_list_j = tmp_s_feat[j][mask_j == 1]  # [n2, c]
                vt = s_x[j]  # c
                if positive_list_j.shape[0] < 1 or negative_list_j.shape[0] < 1:
                    temp_loss = torch.tensor([0.0], device='cuda')
                    count = count - 1
                else:
                    temp_loss = self.hard_triplet_dig(vt, positive_list_j, negative_list_j)
                shot_loss = shot_loss + temp_loss
            s_triplet_loss = s_triplet_loss + shot_loss / max(count, 1)

        s_triplet_loss = s_triplet_loss / shot

        for i in range(b):
            vt = q_x[i]  # c
            positive = query_hard_fg_pro[i]  # c
            negative = query_bg_pro[i]  # c

            edu_distance_pos = F.pairwise_distance(F.normalize(vt, p=2, dim=-1), F.normalize(positive, p=2, dim=-1))
            edu_distance_neg = F.pairwise_distance(F.normalize(vt, p=2, dim=-1), F.normalize(negative, p=2, dim=-1))
            temp_loss = max(0, 0.5 + edu_distance_pos - edu_distance_neg)
            q_triplet_loss = q_triplet_loss + temp_loss

        return s_triplet_loss + q_triplet_loss / b

    def hard_triplet_dig(self, vt, positive, negative):
        """
            anchor: [c]
            positive: [n1, c]
            negative: [n2, c]
        """
        edu_distance_pos = F.pairwise_distance(F.normalize(vt, p=2, dim=-1),
                                               F.normalize(positive, p=2, dim=-1))
        edu_distance_neg = F.pairwise_distance(F.normalize(vt, p=2, dim=-1),
                                               torch.mean(F.normalize(negative, p=2, dim=-1), dim=0, keepdim=True))
        neg_val, _ = edu_distance_neg.sort()
        pos_val, _ = edu_distance_pos.sort()
        triplet_loss = max(0, 0.5 + pos_val[-1] - neg_val[0])  # 0.5
        return triplet_loss
