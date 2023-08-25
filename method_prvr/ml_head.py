from turtle import forward
import torch
import torch.nn as nn
import numpy as np
# import random

from utils.model_utils import generate_gauss_weight, get_props_from_indices
from modules.ml_head import ML_Head as BaseML_Head

class ML_Head(BaseML_Head):
    def __init__(self, config):
        super(ML_Head, self).__init__(config)
        # self.config = config

        # self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)


    def forward(self, key_clip_indices, clip_level_query_context_scores, query_labels, label_dict, video_feat, video_feat_mask, words_len, words_feat, words_weights, modular_roberta_feat, epoch=1, train=True):
        
        sim_based_res = self.sim_based_results(key_clip_indices, clip_level_query_context_scores, query_labels, video_feat, video_feat_mask, modular_roberta_feat, epoch, train=True)

        rec_based_res = self.get_rec_based_results(video_feat, video_feat_mask, words_feat, words_len, words_weights, sim_based_res['center'], sim_based_res['width'], query_labels, label_dict, sim_based_res['gauss_guided_q2vscore'])

        return sim_based_res, rec_based_res

    # no-cross cpl head
    def sim_based_results(self, key_clip_indices, clip_level_query_context_scores, query_labels, video_feat, video_feat_mask, modular_roberta_feat, epoch=1, train=True):

        video_feat = video_feat[query_labels]
        video_feat_mask = video_feat_mask[query_labels]

        num_video, max_frames, _ = video_feat.shape # video_feat.shape[0]=words_feat.shape[0]

        num_query, d = modular_roberta_feat.shape
        props_topk = self.config.props_topk
        ###########################
        # generate gaussian mask
        key_clip_center_prop, _ = get_props_from_indices(key_clip_indices, self.gauss_center, self.gauss_width)
        # selected_center_prop = torch.diag(key_clip_center_prop) # [nq, ] -- only matched
        selected_center_prop = key_clip_center_prop[[i for i in range(key_clip_center_prop.shape[0])], query_labels] # [nq, ]
        gauss_center = selected_center_prop.unsqueeze(-1).expand(num_query, self.num_props).reshape(-1)
        gauss_width = torch.linspace(self.width_lower_bound, self.width_upper_bound, steps=self.num_props).unsqueeze(0).expand(num_query, -1).reshape(-1).to(video_feat.device)

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
        neg_1_weight_l, neg_2_weight_l = self.negative_proposal_mining_v2(max_frames, gauss_center, gauss_width, epoch) # v2
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