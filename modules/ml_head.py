from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import random

from utils.model_utils import get_gauss_props_from_clip_indices, generate_gauss_weight, get_center_based_props, get_sliding_window_based_props, get_props_from_indices
import torch.nn.functional as F

class ML_Head(nn.Module):
    def __init__(self, config):
        super(ML_Head, self).__init__()
        self.config = config

        self.num_props = config.num_props

        self.mapping_linear = nn.ModuleList([nn.Linear(config.hidden_size, out_features=config.hidden_size, bias=False)
                                             for i in range(2)])
        self.modular_video_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)

        # self.max_epoch = config.max_epoch
        self.sigma = config.sigma # cpl
        self.gamma = config.gamma
        self.proposal_method = config.proposal_method
        # self.dropout = config.input_drop

        # infer
        self.num_gauss_center = config.num_gauss_center
        self.num_gauss_width = config.num_gauss_width

        self.width_lower_bound = config.width_lower_bound
        self.width_upper_bound = config.width_upper_bound

        self.map_size = config.map_size
        self.window_size_inc = config.window_size_inc
        # proposal generation
        if config.clip_proposal == 'sliding_window_based':
            self.gauss_center, self.gauss_width = get_sliding_window_based_props(self.map_size, self.window_size_inc, self.width_lower_bound, self.width_upper_bound)
        else:
            self.gauss_center, self.gauss_width = get_center_based_props(self.num_gauss_center, self.num_gauss_width, self.width_lower_bound, self.width_upper_bound)


    def froze_param(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def unfroze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True


    def forward(self, key_clip_indices, clip_level_query_context_scores, query_labels, label_dict, video_feat, video_feat_mask, words_len, words_feat, words_weights, modular_roberta_feat, epoch=1, train=True):
        
        sim_based_res = self.sim_based_results(key_clip_indices, clip_level_query_context_scores, query_labels, video_feat, video_feat_mask, modular_roberta_feat, epoch, train=True)

        return sim_based_res

    # no-cross cpl head
    def sim_based_results(self, key_clip_indices, clip_level_query_context_scores, video_feat, video_feat_mask, modular_roberta_feat, epoch=1, train=True):

        num_video, max_frames, _ = video_feat.shape # video_feat.shape[0]=words_feat.shape[0]

        num_query, d = modular_roberta_feat.shape
        props_topk = self.config.props_topk
        ###########################
        # generate gaussian mask
        if self.proposal_method == 'MS_topk_proposal':
            # get topk multi-scale clip and its information
            q2v_topk_scores, q2v_topk_indices = torch.topk(clip_level_query_context_scores.permute(0,2,1), self.num_props, dim=-1) # (nq, nv, num_props)
            selected_q2v_topk_indices = torch.diag(q2v_topk_indices) # (nq, num_props)
            gauss_center, gauss_width = get_props_from_indices(selected_q2v_topk_indices, self.num_gauss_center, self.num_gauss_width)
            gauss_center = gauss_center.reshape(-1)
            gauss_width = gauss_width.reshape(-1)
        elif self.proposal_method == 'handcraft_width':
            key_clip_center_prop, _ = get_props_from_indices(key_clip_indices, self.gauss_center, self.gauss_width)
            
            selected_center_prop = torch.diag(key_clip_center_prop) # [nq, ] -- only matched
            gauss_center = selected_center_prop.unsqueeze(-1).expand(num_query, self.num_props).reshape(-1)
                                                                                          
            gauss_width = torch.linspace(self.width_lower_bound, self.width_upper_bound, steps=self.num_props).unsqueeze(0).expand(num_query, -1).reshape(-1).to(video_feat.device)
        elif self.proposal_method == 'multi_center_HC_width':
            # choose multi top center in clip branch's proposal
            nq, nps, nv = clip_level_query_context_scores.shape
            idx = [i for i in range(nq)]
            matched_q2c_props_scores = clip_level_query_context_scores.permute(0,2,1)[idx, idx]
            q2c_props_scores = matched_q2c_props_scores.reshape(nq, self.num_gauss_center, self.num_gauss_width)
            # sum or max strategy to choose topk props for every gaussian center (sum strategy only for didemo)
            q2c_center_scores, q2c_width_indices = q2c_props_scores.max(dim=-1) #TODO: max or sum
            q2c_topk_center_socres, q2c_topk_center_indices = torch.topk(q2c_center_scores, props_topk, dim=-1)
            # q2c_topk_width_indices = torch.gather(q2c_width_indices, dim=-1, index=q2c_topk_center_indices) # for calcing width
            centers = torch.linspace(self.width_lower_bound/2, 1 - self.width_lower_bound/2, steps=self.num_gauss_center).unsqueeze(0).expand(nq, self.num_gauss_center).type_as(video_feat)
            q2c_topk_center = torch.gather(centers, dim=-1, index=q2c_topk_center_indices)

            gauss_center = q2c_topk_center.unsqueeze(-1).expand(nq, props_topk, self.num_props).reshape(-1)
            gauss_width = torch.linspace(self.width_lower_bound, self.width_upper_bound, steps=self.num_props).unsqueeze(0).unsqueeze(0).expand(nq, props_topk, self.num_props).reshape(-1).type_as(video_feat)
        
        elif self.proposal_method == 'CDHW': # carefully designed handcraft_width
            key_clip_center_prop, key_clip_width_prop = get_props_from_indices(key_clip_indices, self.gauss_center, self.gauss_width)
            
            selected_center_prop = torch.diag(key_clip_center_prop) # [nq, ] -- only matched
            gauss_center = selected_center_prop.unsqueeze(-1).expand(num_query, self.num_props).reshape(-1)
                                                                                          
            selected_width_prop = torch.diag(key_clip_width_prop)
            gauss_width = []
            for width in selected_width_prop:
                if width - 0.2 < 0.03125:
                    left_edge = max(0.03125, width - 0.2)
                    right_edge = left_edge + 0.4
                else:
                    right_edge = min(width + 0.2, 1)
                    left_edge = right_edge - 0.4
                width_prop = torch.linspace(left_edge, right_edge, steps=self.num_props)
                gauss_width.append(width_prop)
            gauss_width = torch.cat(gauss_width, dim=0).to(video_feat.device)

        else:
            raise Exception('proposal method error')

        # positive proposal
        gauss_weight_l = generate_gauss_weight(max_frames, gauss_center, gauss_width, sigma=self.sigma)
        props_sim_scores = self.gauss_guided_q2v_similarity(gauss_weight_l, modular_roberta_feat, video_feat, self.num_props * props_topk) # gauss_weighted pooling
        gauss_guided_q2vscore, ggq2v_indices = props_sim_scores.max(dim=-1) # alternative: topk mean
        # unnormed
        global_props_vid_feat = self.gauss_weighted_pooling(video_feat, None, gauss_weight_l, self.num_props * props_topk).view(num_query, self.num_props * props_topk, -1)
        props_sim_scores_ = torch.einsum("nd,mpd->nmp", modular_roberta_feat, global_props_vid_feat)
        gauss_guided_q2vscore_, ggq2v_indices_ = props_sim_scores_.max(dim=-1) # alternative: topk mean

        # negative mining
        # added for frame branch intra-video contrastive learning
        # neg_1_weight_l, neg_2_weight_l = self.negative_proposal_mining(max_frames, gauss_center, gauss_width, epoch)
        neg_1_weight_l, neg_2_weight_l = self.negative_proposal_mining(max_frames, gauss_center, gauss_width, epoch) # v2
        neg1_gauss_guided_q2props_score = self.gauss_guided_q2v_similarity(neg_1_weight_l, modular_roberta_feat, video_feat, self.num_props * props_topk)
        neg2_gauss_guided_q2props_score = self.gauss_guided_q2v_similarity(neg_2_weight_l, modular_roberta_feat, video_feat, self.num_props * props_topk)
        neg1_selected_ggq2v_score = torch.gather(neg1_gauss_guided_q2props_score, dim=-1, index=ggq2v_indices.unsqueeze(-1)).squeeze(-1)
        neg2_selected_ggq2v_score = torch.gather(neg2_gauss_guided_q2props_score, dim=-1, index=ggq2v_indices.unsqueeze(-1)).squeeze(-1)
        # hard negative: the whole video
        ref_weight = torch.ones(num_video, max_frames).type_as(video_feat)
        hard_neg_score = self.gauss_guided_q2v_similarity(ref_weight, modular_roberta_feat, video_feat, 1).squeeze(-1)
        # 1 - pos
        easy_neg_weight = 1 - gauss_weight_l
        easy_neg_weight = easy_neg_weight / easy_neg_weight.max(dim=-1, keepdim=True)[0]
        easy_neg_props_score = self.gauss_guided_q2v_similarity(easy_neg_weight, modular_roberta_feat, video_feat, self.num_props * props_topk)
        easy_neg_score = torch.gather(easy_neg_props_score, dim=-1, index=ggq2v_indices.unsqueeze(-1)).squeeze(-1)
        ###############
        # hardest neg: the worst proposal of the key-center-guided proposals
        hardest_neg_score, _ = props_sim_scores_.min(dim=-1)


        return {
            'width': gauss_width,
            'center': gauss_center,
            'gauss_guided_q2vscore': gauss_guided_q2vscore,
            'gauss_guided_q2vscore_': gauss_guided_q2vscore_,
            'props_sim_scores': props_sim_scores,
            'neg1_selected_ggq2v_score': neg1_selected_ggq2v_score,
            'neg2_selected_ggq2v_score': neg2_selected_ggq2v_score,
            "hard_neg_score": hard_neg_score,
            "easy_neg_score": easy_neg_score,
            "hardest_neg_score": hardest_neg_score,

        }


    ########################
    # prvr inference # deprecated version
    def get_moment_level_inference_results(self, video_feat, video_feat_mask, modular_roberta_feat):
        num_video, max_ctx_len, _ = video_feat.shape
        
        self.gauss_center = self.gauss_center.to(video_feat.device)
        self.gauss_width = self.gauss_width.to(video_feat.device)

        # calc similarity score, using gaussian weight guided pooling
        gauss_weight = generate_gauss_weight(max_ctx_len, self.gauss_center, self.gauss_width, sigma=self.sigma)
        props_sim_scores = []
        # normalize query feat
        modular_roberta_feat = F.normalize(modular_roberta_feat, dim=-1)

        #########################
        # no-loop version, may require more memory
        gauss_weight = (gauss_weight + 1e-10) / gauss_weight.sum(dim=-1, keepdim=True)
        gauss_weight = gauss_weight.unsqueeze(0).expand(num_video, -1, -1) # [nv, num_props, max_ctx_len]
        gauss_guided_global_vid_feat = torch.bmm(gauss_weight, video_feat) # [nv, num_props, d]
        gauss_guided_global_vid_feat = F.normalize(gauss_guided_global_vid_feat, dim=-1) # normalize
        props_sim_scores = torch.matmul(gauss_guided_global_vid_feat, modular_roberta_feat.t()).permute(2, 0, 1) #[nq, nv, np]


        sim_scores = props_sim_scores.max(dim=-1)[0]

        return sim_scores, props_sim_scores
    
    # vcmr inference : deprecated version
    def get_vcmr_inference_results(self, video_feat, video_feat_mask, modular_roberta_feat, num_props_for_each_video=10):
        num_video, max_ctx_len, _ = video_feat.shape
        self.gauss_center = self.gauss_center.to(video_feat.device)
        self.gauss_width = self.gauss_width.to(video_feat.device)

        # calc similarity score, using gaussian weight guided pooling
        gauss_weight = generate_gauss_weight(max_ctx_len, self.gauss_center, self.gauss_width, sigma=self.sigma)

        # normalize query feat
        modular_roberta_feat = F.normalize(modular_roberta_feat, dim=-1)

        
        #########################
        # no-loop version, may require more memory
        gauss_weight = (gauss_weight + 1e-10) / gauss_weight.sum(dim=-1, keepdim=True)
        gauss_weight = gauss_weight.unsqueeze(0).expand(num_video, -1, -1) # [nv, num_props, max_ctx_len]
        gauss_guided_global_vid_feat = torch.bmm(gauss_weight, video_feat) # [nv, num_props, d]
        gauss_guided_global_vid_feat = F.normalize(gauss_guided_global_vid_feat, dim=-1) # normalize
        props_sim_scores = torch.matmul(gauss_guided_global_vid_feat, modular_roberta_feat.t()).permute(2, 0, 1) #[nq, nv, np]

        # query-to-video sim scores
        sim_scores = props_sim_scores.max(dim=-1)[0] # query-video sim scores

        #########################
        # select top-k proposals from all proposals
        num_props_for_each_video = min(num_props_for_each_video, props_sim_scores.shape[-1])
        # print("***", props_sim_scores.shape, props_sim_scores.device, "***")
        selected_props_scores, selected_props_indices = torch.topk(props_sim_scores, num_props_for_each_video, dim=-1)
        num_query, num_video, npev = selected_props_scores.shape
        flatted_selected_props_indices = selected_props_indices.view(num_query * num_video, npev)
        # flatted_gauss_center, flatted_gauss_width = get_gauss_props_from_clip_indices(flatted_selected_props_indices, self.num_gauss_center, self.num_gauss_width, self.width_lower_bound, self.width_upper_bound)
        flatted_gauss_center, flatted_gauss_width = get_props_from_indices(flatted_selected_props_indices, self.gauss_center, self.gauss_width)
        
        flatted_selected_props = torch.stack([torch.clamp(flatted_gauss_center - flatted_gauss_width/2, min=0), 
                                                torch.clamp(flatted_gauss_center + flatted_gauss_width/2, max=1)], dim=-1)
        selected_props = flatted_selected_props.view(num_query, num_video, npev, 2)

        return selected_props, selected_props_scores, sim_scores

    
    def negative_proposal_mining(self, props_len, center, width, epoch=1):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma/2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w/w1*torch.exp(-(pos-c)**2/(2*w1**2))
            return y1/y1.max(dim=-1, keepdim=True)[0]

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center-width/2, min=0)
        # left_center = left_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5
        left_center = torch.zeros_like(center)
        right_width = torch.clamp(1-center-width/2, min=0)
        # right_center = 1 - right_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5
        right_center = torch.ones_like(center)

        left_neg_weight = Gauss(weight, left_width, left_center)
        right_neg_weight = Gauss(weight, right_width, right_center)

        return left_neg_weight, right_neg_weight
    
    # correct version
    def negative_proposal_mining_new(self, props_len, center, width, epoch=1):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma/2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w/w1*torch.exp(-(pos-c)**2/(2*w1**2))
            return y1/y1.max(dim=-1, keepdim=True)[0]

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center-width/2, min=0)
        left_center = left_width * 0.5
        # left_center = torch.zeros_like(center)
        right_width = torch.clamp(1-center-width/2, min=0)
        # right_center = 1 - right_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5
        right_center = 1 - right_width * 0.5
        right_center = torch.ones_like(center)

        left_neg_weight = Gauss(weight, left_width, left_center)
        right_neg_weight = Gauss(weight, right_width, right_center)

        return left_neg_weight, right_neg_weight

    

    def gauss_guided_q2v_similarity(self, gauss_weight, modular_roberta_feat, video_feat, num_props):
        '''
        using gauss weight to pool the video feat to get global video feat, 
        calculating cosine similarity between global video feat and query feat
        '''
        num_video, _, _ = video_feat.shape
        # # ggp:
        global_props_vid_feat = self.gauss_weighted_pooling(video_feat, None, gauss_weight, num_props).view(num_video, num_props, -1)

        modular_roberta_feat = F.normalize(modular_roberta_feat, dim=-1)
        global_props_vid_feat = F.normalize(global_props_vid_feat, dim=-1)

        props_sim_scores = torch.einsum("nd,mpd->nmp", modular_roberta_feat, global_props_vid_feat)

        return props_sim_scores
    
    def gauss_weighted_pooling(self, frame_feat, frame_mask, gauss_weight, num_props):
        nv, lv, d = frame_feat.shape
        if frame_feat.shape[0] != gauss_weight.shape[0]:
            # frame_feat = torch.repeat_interleave(frame_feat, num_props, dim=0) # old, slow
            frame_feat = frame_feat.unsqueeze(1).expand(nv, num_props, lv, d).reshape(nv*num_props, lv, d)
            # props_vid_mask = torch.repeat_interleave(frame_mask, num_props, dim=0)
        gauss_weight = (gauss_weight + 1e-10) / gauss_weight.sum(dim=-1, keepdim=True)# normalize
        # global_props_vid_feat = torch.einsum("bl,bld->bd", gauss_weight, frame_feat) # old, slower
        global_props_vid_feat = torch.bmm(gauss_weight.unsqueeze(1), frame_feat).squeeze(1)
        return global_props_vid_feat


    def get_self_atten_q2vscores(self, frame_feat, frame_feat_mask, modular_query):
        modular_atten_scores = self.modular_video_mapping(frame_feat)
        modular_atten_scores = F.softmax(mask_logits(modular_atten_scores, frame_feat_mask.unsqueeze(2)), dim=1)
        # modular_frame_feat = torch.einsum("blm,bld->bmd", modular_atten_scores, frame_feat).squeeze(1) # old
        modular_frame_feat = torch.bmm(modular_atten_scores.transpose(2,1), frame_feat).squeeze(1)
        frame_scale_scores = torch.matmul(F.normalize(modular_query, dim=-1),
                                              F.normalize(modular_frame_feat, dim=-1).t())
        frame_scale_scores_ = torch.matmul(modular_query, modular_frame_feat.t())  # unnormalized

        return frame_scale_scores, frame_scale_scores_
    

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)

def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().to(x.device))
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask