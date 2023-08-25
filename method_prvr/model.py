import torch
import torch.nn as nn
import torch.nn.functional as F
from method_prvr.ml_head import ML_Head

from modules.model import GGCLNet as BaseModel


class GGCLNet(BaseModel):
    def __init__(self, config):
        
        super(GGCLNet, self).__init__(config)

        # ml head
        self.ml_head = ML_Head(config) # changed order, influence?
        # self.config = config
        self.reset_parameters() # two times?

    def forward(self, clip_video_feat, frame_video_feat, frame_video_mask, query_feat, query_mask, query_labels, epoch=1): # with rec: , words_id, words_feat, words_len, words_weights; per video:, query_labels


        encoded_frame_feat, vid_proposal_feat, encoded_clip_feat = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask)
        nq, np, nv = vid_proposal_feat.shape
        clip_scale_scores, frame_scale_scores, clip_scale_scores_, frame_scale_scores_, var4ml \
            = self.get_pred_from_raw_query(
            query_feat, query_mask, vid_proposal_feat, encoded_frame_feat, frame_video_mask,# cross=False,
            return_query_feats=True, query_labels=query_labels)

        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        clip_nce_loss = 0.02 * self.clip_nce_criterion(query_labels, label_dict, clip_scale_scores_) # per-video
        clip_trip_loss = self.get_clip_triplet_loss(clip_scale_scores, query_labels) # per video
        clip_loss = clip_nce_loss + clip_trip_loss

        # attention-based frame loss
        atten_nce_loss = atten_trip_loss = 0
        atten_nce_loss = 0.04 * self.video_nce_criterion(frame_scale_scores_)
        atten_trip_loss = self.get_frame_trip_loss(frame_scale_scores)
        atten_loss = atten_nce_loss + atten_trip_loss

        # clip scale intra-video trip loss
        clip_intra_trip_loss = 0
        intra_margin = self.config.intra_margin
        intra_margin2 = intra_margin - 0.1

        alpha1 = self.config.alpha1
        alpha2 = self.config.alpha2
        alpha3 = self.config.alpha3

        inter_video_trip_loss = 0
        intra_video_trip_loss = 0

        # moment localization
        modular_query = var4ml['video_query'].detach() # detach or not
        sim_based_results = self.ml_head.sim_based_results(var4ml['key_clip_indices'], var4ml['q2c_props_scores'], query_labels, 
                                                encoded_frame_feat, frame_video_mask, modular_query, epoch) # var4ml['video_query'].detach()
        frame_selected_scores = sim_based_results['gauss_guided_q2vscore']
        unnormed_frame_selected_scores = sim_based_results['gauss_guided_q2vscore_']
        # inter-vid loss
        inter_video_trip_loss = self.get_frame_trip_loss(frame_selected_scores)
        inter_video_nce_loss=0
        inter_video_nce_loss = 0.02 * self.video_nce_criterion(unnormed_frame_selected_scores)
        inter_video_CL_loss = inter_video_nce_loss + inter_video_trip_loss

        # frame scale intra-video trip loss
        frame_intra_trip_loss = 0
        idx = [i for i in range(nq)]
        matched_neg1_score = torch.diag(sim_based_results['neg1_selected_ggq2v_score']) #[idx, idx]
        matched_neg2_score = torch.diag(sim_based_results['neg2_selected_ggq2v_score']) 
        matched_hard_neg_score = torch.diag(sim_based_results['hard_neg_score'])
        matched_easy_neg_score = torch.diag(sim_based_results['easy_neg_score'])
        matched_hardest_neg_score = torch.diag(sim_based_results['hardest_neg_score'])
        matched_frame_score = torch.diag(frame_selected_scores)
        # side negative:
        _, frame_intra_trip_loss_1 = self.vector_trip_loss(matched_frame_score, matched_neg1_score, intra_margin)
        _, frame_intra_trip_loss_2 = self.vector_trip_loss(matched_frame_score, matched_neg2_score, intra_margin)
        # hard negative:
        _, frame_intra_trip_loss_hard = self.vector_trip_loss(matched_frame_score, matched_hard_neg_score, 0.1)
        # easy negative:
        _, frame_intra_trip_loss_easy = self.vector_trip_loss(matched_frame_score, matched_easy_neg_score, 0.1)
        # hardest negative:
        _, frame_intra_trip_loss_hardest = self.vector_trip_loss(matched_frame_score, matched_hardest_neg_score, 0.1)
        
        frame_intra_trip_loss = frame_intra_trip_loss_1 + frame_intra_trip_loss_2 + frame_intra_trip_loss_hard


        # reconstruction based learning
        pos_rec_loss = 0
        rec_neg_vid_trip_loss = 0
        rec_neg_query_trip_loss = 0
        pos_sl_rec_loss = 0
        intra_vid_rec_cl_loss = 0
        kd_kl_loss = 0
        
        loss = clip_loss + 1*inter_video_trip_loss + atten_loss

        return loss, {"loss_overall": float(loss), 'clip_nce_loss': clip_nce_loss,
                      'clip_trip_loss': clip_trip_loss,
                    #   'frame_nce_loss': frame_nce_loss, 'frame_trip_loss': frame_trip_loss,
                      'intra_vid_trip_loss': intra_video_trip_loss, 'inter_video_trip_loss': inter_video_trip_loss,
                       'clip_intra_trip_loss': clip_intra_trip_loss, 'inter_video_nce_loss': inter_video_nce_loss,
                       'frame_intra_trip_loss': frame_intra_trip_loss, "frame_intra_trip_loss_hardest": frame_intra_trip_loss_hardest,
                      'pos_rec_loss':pos_rec_loss, 'rec_neg_vid_trip_loss': rec_neg_vid_trip_loss, 
                      'rec_neg_query_trip_loss': rec_neg_query_trip_loss, 'pos_sl_rec_loss': pos_sl_rec_loss,
                      'intra_vid_rec_cl_loss': intra_vid_rec_cl_loss, "kd_kl_loss": kd_kl_loss}


    # ori verison, per-video
    def key_clip_guided_attention(self, frame_feat, proposal_feat, feat_mask, max_index, query_labels):
        selected_max_index = max_index[[i for i in range(max_index.shape[0])], query_labels]

        expand_frame_feat = frame_feat[query_labels]

        expand_proposal_feat = proposal_feat[query_labels]

        key = self.mapping_linear[0](expand_frame_feat)
        query = expand_proposal_feat[[i for i in range(key.shape[0])], selected_max_index, :].unsqueeze(-1)
        value = self.mapping_linear[1](expand_frame_feat)

        if feat_mask is not None:
            expand_feat_mask = feat_mask[query_labels]
            scores = torch.bmm(key, query).squeeze()
            masked_scores = scores.masked_fill(expand_feat_mask.eq(0), -1e9).unsqueeze(1)
            masked_scores = nn.Softmax(dim=-1)(masked_scores)
            attention_feat = torch.bmm(masked_scores, value).squeeze()
        else:
            scores = nn.Softmax(dim=-1)(torch.bmm(key, query).transpose(1, 2))
            attention_feat = torch.bmm(scores, value).squeeze()

        return attention_feat

    
    def get_pred_from_raw_query(self, query_feat, query_mask, 
                                video_proposal_feat=None,
                                video_feat=None,
                                video_feat_mask=None,
                                return_query_feats=False,
                                query_labels=None):


        video_query = self.encode_query(query_feat, query_mask)

        # get clip-level retrieval scores

        clip_scale_scores, key_clip_indices, q2c_props_scores = self.get_clip_scale_scores(
            video_query, video_proposal_feat)
        
        # for ml_head:
        var4ml = dict()
        var4ml['key_clip_indices'] = key_clip_indices
        var4ml['video_query'] = video_query
        var4ml['q2c_props_scores'] = q2c_props_scores

        if return_query_feats:
            frame_scale_feat = self.key_clip_guided_attention(video_feat, video_proposal_feat, video_feat_mask, 
                                                          key_clip_indices, query_labels) # , query_labels
            frame_scale_scores = torch.matmul(F.normalize(video_query, dim=-1), # video_query.detach()
                                              F.normalize(frame_scale_feat, dim=-1).t())
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            frame_scale_scores_ = torch.matmul(video_query, frame_scale_feat.t()) # video_query.detach()


            return clip_scale_scores, frame_scale_scores, clip_scale_scores_,frame_scale_scores_, var4ml

        else:
            frame_scale_feat = self.key_clip_guided_attention_in_inference(video_feat, video_proposal_feat, video_feat_mask,
                                                                       key_clip_indices).to(video_query.device)
            frame_scales_cores_ = torch.mul(F.normalize(frame_scale_feat, dim=-1),
                                            F.normalize(video_query, dim=-1).unsqueeze(0))
            frame_scale_scores = torch.sum(frame_scales_cores_, dim=-1).transpose(1, 0)

            return clip_scale_scores, frame_scale_scores, var4ml
        
