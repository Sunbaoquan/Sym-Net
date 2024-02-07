import torch
class Correlation:
    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        # query_feat: b,c,h,w
        # support_feat: [[(c, N) x b] x nfeatures]
        eps = 1e-5
        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            b, c, h, w = query_feat.shape  # b,c,h,w
            corrI = []
            for j in range(b):  # b
                queryIJ = query_feat[j].flatten(start_dim=1)  # c,hw
                queryIJNorm = queryIJ / (queryIJ.norm(dim=0, p=2, keepdim=True) + eps)
                supIJ = support_feat[j]  # c, N
                supIJNorm = supIJ / (supIJ.norm(dim=0, p=2, keepdim=True) + eps)
                corr = (queryIJNorm.permute(1, 0)).matmul(supIJNorm)  # hw N
                corr = corr.clamp(min=0)
                corr = corr.mean(dim=1, keepdim=True)  # hw, 1
                corr = (corr.permute(1, 0)).unsqueeze(0)  # 1,1,hw
                corrI.append(corr)  # [(1,1,hw) x b]
            corrI = torch.cat(corrI, dim=0)  # b,1,hw
            corrI = corrI.view((b, -1, h, w))  # b,1,h,w
            corrs.append(corrI)  # b,1,h,w x nfeatures

        corr_l4 = torch.cat(corrs[-stack_ids[0]:], dim=1).contiguous()  # b,n,h,w
        corr_l3 = torch.cat(corrs[-stack_ids[1]:-stack_ids[0]], dim=1).contiguous()
        corr_l2 = torch.cat(corrs[-stack_ids[2]:-stack_ids[1]], dim=1).contiguous()
        return [corr_l4, corr_l3, corr_l2]
