import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from modules.model_components import BertAttention, LinearLayer, BertSelfAttention, TrainablePositionalEncoding, GlobalPool1D, GlobalConv1D
from modules.model_components import clip_nce, frame_nce
from utils.model_utils import generate_gauss_weight, get_center_based_props, get_sliding_window_based_props, get_props_from_indices, generate_mean_weight
from modules.ml_head import ML_Head



class GGCLNet(nn.Module):
    def __init__(self, config):
        super(GGCLNet, self).__init__()
        self.config = config

        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
        #
        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        #
        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))


        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.clip_encoder = copy.deepcopy(self.query_encoder)

        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
        self.frame_encoder = copy.deepcopy(self.query_encoder)

        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)


        self.pool_layers = nn.ModuleList([nn.Identity()]
                                         + [nn.AvgPool1d(i, stride=1) for i in range(2, config.map_size + 1)]
                                         )

        self.mapping_linear = nn.ModuleList([nn.Linear(config.hidden_size, out_features=config.hidden_size, bias=False)
                                             for i in range(2)])

        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.video_nce_criterion = frame_nce(reduction='mean')

        # ml head
        self.ml_head = None #ML_Head(config) # TODO: inited in specific model

        # global sample
        self.global_pool1d = GlobalPool1D(config.pooling_ksize, config.pooling_stride)
        self.global_conv1d = GlobalConv1D(config.visual_input_size, config.conv_ksize, config.conv_stride, out_channel=1)
        # not need
        # self.global_conv1d2 = GlobalConv1D(config.visual_input_size, config.global_ksize, 1, out_channel=1)

        # gauss weighted multi-scal clip feat
        self.num_gauss_center = config.num_gauss_center # 32
        self.num_gauss_width = config.num_gauss_width # 10
        self.sigma = config.sigma # 9
        self.width_lower_bound = config.width_lower_bound
        self.width_upper_bound = config.width_upper_bound
        self.map_size = config.map_size
        self.window_size_inc = config.window_size_inc

        # proposal generation
        if config.clip_proposal == 'sliding_window_based':
            self.gauss_center, self.gauss_width = get_sliding_window_based_props(self.map_size, self.window_size_inc, self.width_lower_bound, self.width_upper_bound)
        else:
            self.gauss_center, self.gauss_width = get_center_based_props(self.num_gauss_center, self.num_gauss_width, self.width_lower_bound, self.width_upper_bound)


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

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size



    def forward(self, clip_video_feat, frame_video_feat, frame_video_mask, query_feat, query_mask, epoch): # with rec: , words_id, words_feat, words_len, words_weights; per video:, query_labels


        encoded_frame_feat, vid_proposal_feat, encoded_clip_feat = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask)
        nq, np, nv = vid_proposal_feat.shape
        clip_scale_scores, frame_scale_scores, clip_scale_scores_, frame_scale_scores_, var4ml \
            = self.get_pred_from_raw_query(
            query_feat, query_mask, vid_proposal_feat, encoded_clip_feat, #, frame_video_mask,# cross=False,
            return_query_feats=True)

        rt = 0.04 # 0.02
        clip_nce_loss = rt * self.video_nce_criterion(clip_scale_scores_)
        clip_trip_loss = self.get_frame_trip_loss(clip_scale_scores)
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
        sim_based_results = self.ml_head.sim_based_results(var4ml['key_clip_indices'], var4ml['q2c_props_scores'], # query_labels, 
                                                encoded_frame_feat, frame_video_mask, var4ml['video_query'], epoch) # var4ml['video_query'].detach()
        frame_selected_scores = sim_based_results['gauss_guided_q2vscore']
        unnormed_frame_selected_scores = sim_based_results['gauss_guided_q2vscore_']
        # inter-vid loss
        inter_video_trip_loss = self.get_frame_trip_loss(frame_selected_scores)
        inter_video_nce_loss=0
        inter_video_nce_loss = rt * self.video_nce_criterion(unnormed_frame_selected_scores)
        inter_video_CL_loss = inter_video_nce_loss + inter_video_trip_loss

        # frame scale intra-video trip loss
        frame_intra_trip_loss = 0
        idx = [i for i in range(nq)]
        matched_neg1_score = torch.diag(sim_based_results['neg1_selected_ggq2v_score']) #[idx, idx]
        matched_neg2_score = torch.diag(sim_based_results['neg2_selected_ggq2v_score']) 
        matched_hard_neg_score = torch.diag(sim_based_results['hard_neg_score'])
        # matched_easy_neg_score = torch.diag(sim_based_results['easy_neg_score'])
        # matched_hardest_neg_score = torch.diag(sim_based_results['hardest_neg_score'])
        matched_frame_score = torch.diag(frame_selected_scores)
        # side negative:
        _, frame_intra_trip_loss_1 = self.vector_trip_loss(matched_frame_score, matched_neg1_score, intra_margin)
        _, frame_intra_trip_loss_2 = self.vector_trip_loss(matched_frame_score, matched_neg2_score, intra_margin)
        # hard negative:
        _, frame_intra_trip_loss_hard = self.vector_trip_loss(matched_frame_score, matched_hard_neg_score, 0.1)

        frame_intra_trip_loss = frame_intra_trip_loss_1 + frame_intra_trip_loss_2 + frame_intra_trip_loss_hard



        # reconstruction based learning
        pos_rec_loss = 0
        rec_neg_vid_trip_loss = 0
        rec_neg_query_trip_loss = 0
        pos_sl_rec_loss = 0
        intra_vid_rec_cl_loss = 0
        kd_kl_loss = 0
        

        loss = clip_loss + inter_video_CL_loss + frame_intra_trip_loss

        return loss, {"loss_overall": float(loss), 'clip_nce_loss': clip_nce_loss,
                      'clip_trip_loss': clip_trip_loss,
                    #   'frame_nce_loss': frame_nce_loss, 'frame_trip_loss': frame_trip_loss,
                      'intra_vid_trip_loss': intra_video_trip_loss, 'inter_video_trip_loss': inter_video_trip_loss,
                       'clip_intra_trip_loss': clip_intra_trip_loss, 'inter_video_nce_loss': inter_video_nce_loss,
                       'frame_intra_trip_loss': frame_intra_trip_loss, 
                      'pos_rec_loss':pos_rec_loss, 'rec_neg_vid_trip_loss': rec_neg_vid_trip_loss, 
                      'rec_neg_query_trip_loss': rec_neg_query_trip_loss, 'pos_sl_rec_loss': pos_sl_rec_loss,
                      'intra_vid_rec_cl_loss': intra_vid_rec_cl_loss, "kd_kl_loss": kd_kl_loss}
    

    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        video_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 1

        return video_query

    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None, return_frame_props_feat=False):
        if self.config.global_sample:
            # pooling then conv
            glob_vid_feat, glob_vid_mask = self.global_pool1d(frame_video_feat, video_mask) # changed
            glob_vid_feat, glob_vid_mask = self.global_conv1d(glob_vid_feat, glob_vid_mask) # uncomment to use
            
            # encode after downsampling
            encoded_clip_feat = self.encode_input(glob_vid_feat, glob_vid_mask, self.clip_input_proj, self.clip_encoder,
                                                    self.clip_pos_embed)
            
        else:
            # ori
            encoded_clip_feat = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                                self.clip_pos_embed)
            # encoded_clip_feat = self.encode_input(frame_video_feat, video_mask, self.clip_input_proj, self.clip_encoder,
            #                                     self.clip_pos_embed)
        
        
        if self.config.shared_trans == 'true':
            frame_encoder = self.clip_encoder
        else:
            frame_encoder = self.frame_encoder
        encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                                frame_encoder, 
                                                self.frame_pos_embed)

        vid_proposal_feat_map = self.gauss_weighted_vid_props_feat(encoded_clip_feat)

        if return_frame_props_feat:
            # frame_props_feat_map = self.encode_feat_map(encoded_frame_feat)
            frame_props_feat_map = self.gauss_weighted_vid_props_feat(encoded_frame_feat)
            return encoded_frame_feat, vid_proposal_feat_map, encoded_clip_feat, frame_props_feat_map
        else:
            return encoded_frame_feat, vid_proposal_feat_map, encoded_clip_feat # , video_mask
    
    def gauss_weighted_vid_props_feat(self, x_feat): # 有问题！会掉点--原因未知
        # pass
        nv, lv, dv = x_feat.shape

        self.gauss_center = self.gauss_center.to(x_feat.device)
        self.gauss_width = self.gauss_width.to(x_feat.device)
        gauss_weight = generate_gauss_weight(lv, self.gauss_center, self.gauss_width, self.sigma) # gauss weight
        # gauss_weight = generate_mean_weight(lv, self.gauss_center, self.gauss_width) # mean weight
        
        gauss_weight = (gauss_weight + 1e-10) / gauss_weight.sum(dim=-1, keepdim=True)
        # num_props = gauss_center.shape
        gauss_weight = gauss_weight.unsqueeze(0).expand(nv, -1, lv)
        proposal_feat_map = torch.bmm(gauss_weight, x_feat)

        return proposal_feat_map


    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        # modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        modular_queries = torch.bmm(modular_attention_scores.transpose(2,1), encoded_query)
        return modular_queries.squeeze()


    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        query_context_scores, indices = torch.max(clip_level_query_context_scores, # alternative: topk mean
                                                  dim=1)  # (N, N) diagonal positions are positive pairs
        return query_context_scores, indices, clip_level_query_context_scores



    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):

        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        query_context_scores, _ = torch.max(query_context_scores, dim=1) # alternative: topk mean

        return query_context_scores


    # new verison, per-query
    def key_clip_guided_attention(self, frame_feat, proposal_feat, feat_mask, max_index):
        selected_max_index = torch.diag(max_index)

        key = self.mapping_linear[0](frame_feat)
        query = proposal_feat[[i for i in range(key.shape[0])], selected_max_index, :].unsqueeze(-1)
        value = self.mapping_linear[1](frame_feat)

        if feat_mask is not None:
            scores = torch.bmm(key, query).squeeze()
            masked_scores = scores.masked_fill(feat_mask.eq(0), -1e9).unsqueeze(1)
            masked_scores = nn.Softmax(dim=-1)(masked_scores)
            attention_feat = torch.bmm(masked_scores, value).squeeze()
        else:
            scores = nn.Softmax(dim=-1)(torch.bmm(key, query).transpose(1, 2))
            attention_feat = torch.bmm(scores, value).squeeze()

        return attention_feat

    def key_clip_guided_attention_in_inference(self, frame_feat, proposal_feat, feat_mask, max_index):
        key = self.mapping_linear[0](frame_feat)
        value = self.mapping_linear[1](frame_feat)
        num_vid = frame_feat.shape[0]

        index = torch.arange(num_vid).unsqueeze(1)
        query = proposal_feat[index, max_index.t()]
        if feat_mask is not None:
            scores = torch.bmm(key, query.transpose(2, 1))
            masked_scores = scores.masked_fill(feat_mask.unsqueeze(-1).eq(0), -1e9)
            masked_scores = nn.Softmax(dim=1)(masked_scores)
            attention_feat = torch.bmm(masked_scores.transpose(1, 2), value)
        else:
            scores = torch.bmm(key, query.transpose(2, 1))
            scores = nn.Softmax(dim=1)(scores)
            attention_feat = torch.bmm(scores.transpose(1, 2), value)

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
            frame_scale_feat = self.key_clip_guided_attention(video_feat, video_proposal_feat, video_feat_mask, # TODO:try detach frame feat, only optimize clip feat
                                                          key_clip_indices) # , query_labels
            frame_scale_scores = torch.matmul(F.normalize(video_query, dim=-1), # video_query.detach()
                                              F.normalize(frame_scale_feat, dim=-1).t())
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            frame_scale_scores_ = torch.matmul(video_query, frame_scale_feat.t()) # video_query.detach()

            # # easy negative proposal scores
            # neg_clip_scale_scores, neg_clip_scale_indices = torch.min(q2c_props_scores, dim=1)

            return clip_scale_scores, frame_scale_scores, clip_scale_scores_,frame_scale_scores_, var4ml

        else:
            frame_scale_feat = self.key_clip_guided_attention_in_inference(video_feat, video_proposal_feat, video_feat_mask,
                                                                       key_clip_indices).to(video_query.device)
            frame_scales_cores_ = torch.mul(F.normalize(frame_scale_feat, dim=-1),
                                            F.normalize(video_query, dim=-1).unsqueeze(0))
            frame_scale_scores = torch.sum(frame_scales_cores_, dim=-1).transpose(1, 0)

            return clip_scale_scores, frame_scale_scores, var4ml # add video_query for wsml_frame_score  
    
    # for event-level inference
    def get_infer_res_from_props_feat(self, modular_query, video_proposal_feat, num_props_for_each_video=10):
        '''
        input:
            video_proposal_feat: [nq, n-clips, nv]
            num_props_for_each_video: int
        output:
            selected_props: [nq, nv, npev, 2]
            q2v_topk_scores: [nq, nv, npev]
        '''
        # video_query = self.encode_query(roberta_feat, roberta_mask) # (N, D)

        # get clip-level retrieval scores
        q2v_scores, key_clip_indices, query_context_props_scores = self.get_clip_scale_scores( 
            modular_query, video_proposal_feat)

        # get topk multi-scale clip and its information
        num_props_for_each_video = min(num_props_for_each_video, query_context_props_scores.shape[1])
        q2v_topk_scores, q2v_topk_indices = torch.topk(query_context_props_scores.permute(0,2,1), num_props_for_each_video, dim=-1) # (nq, nv, npev)
        num_query, num_video, num_props = q2v_topk_scores.shape
        flatted_q2v_topk_indices = q2v_topk_indices.view(num_query*num_video, num_props)
        # flatted_gauss_center, flatted_gauss_width = get_gauss_props_from_clip_indices(flatted_q2v_topk_indices, self.num_gauss_center, self.num_gauss_width, self.width_lower_bound, self.width_upper_bound)
        flatted_gauss_center, flatted_gauss_width = get_props_from_indices(flatted_q2v_topk_indices, self.gauss_center, self.gauss_width)

        # scaling the proposal width
        factor = 0.5

        flatted_selected_props = torch.stack([torch.clamp(flatted_gauss_center - flatted_gauss_width/2, min=0), 
                                                torch.clamp(flatted_gauss_center + flatted_gauss_width/2, max=1)], dim=-1)
        selected_props = flatted_selected_props.view(num_query, num_video, num_props, 2)

        return selected_props, q2v_topk_scores, q2v_scores
    
    def get_mixed_infer_res(self, modular_query, clip_proposal_feat, frame_proposal_feat, num_props_for_each_video=10):
        '''
        input:
            clip_proposal_feat: [nq, n-props, nv]
            frame_proposal_feat: [nq, n-props, nv]
            num_props_for_each_video: int
        output:
            selected_props: [nq, nv, npev, 2]
            q2v_topk_scores: [nq, nv, npev]
        '''
        # get clip-level retrieval scores
        clip_scale_scores, key_clip_indices, clip_scale_props_scores = self.get_clip_scale_scores( 
                                                                                modular_query, clip_proposal_feat)

        # get clip-level retrieval scores
        frame_scale_scores, _, frame_scale_props_scores = self.get_clip_scale_scores( 
                                                                                modular_query, frame_proposal_feat)
        # mixed props scores
        mixed_props_scores = (clip_scale_props_scores.permute(0,2,1) + frame_scale_props_scores.permute(0,2,1)) / 2

        #########################
        # select top-k proposals from all proposals
        num_props_for_each_video = min(num_props_for_each_video, mixed_props_scores.shape[-1])
        selected_props_scores, selected_props_indices = torch.topk(mixed_props_scores, num_props_for_each_video, dim=-1)
        num_query, num_video, npev = selected_props_scores.shape
        flatted_selected_props_indices = selected_props_indices.view(num_query * num_video, npev)
        flatted_gauss_center, flatted_gauss_width = get_props_from_indices(flatted_selected_props_indices, self.gauss_center, self.gauss_width)
        flatted_selected_props = torch.stack([torch.clamp(flatted_gauss_center - flatted_gauss_width/2, min=0), 
                                                torch.clamp(flatted_gauss_center + flatted_gauss_width/2, max=1)], dim=-1)
        selected_props = flatted_selected_props.view(num_query, num_video, npev, 2)

        return selected_props, selected_props_scores, frame_scale_scores


    def get_clip_triplet_loss(self, query_context_scores, labels):
        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])


            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
            if self.config.use_hard_negative:
                sample_neg_pair_scores = neg_pair_scores[0]
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]



            v2t_loss += (self.config.margin + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        t2v_sample_max_idx = min(1 + self.config.hard_pool_size,
                                 t2v_scores.shape[1]) if self.config.use_hard_negative else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]

        t2v_neg_scores = t2v_scores[text_indices, sample_indices]

        t2v_loss = (self.config.margin + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)

    def get_frame_trip_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """

        bsz = len(query_context_scores)

        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx + loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """

        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)

        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

        sample_min_idx = 1  # skip the masked positive

        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) if self.config.use_hard_negative else bsz

        # sample_max_idx = 2

        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]

        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)
    
    ####################################################

    def vector_trip_loss(self, pos_sim_score, neg_sim_score, margin):
        '''
        input:
            pos_sim_score: [n, ]
            neg_sim_score: [n, ]
            margin: float
        '''
        
        trip_loss = torch.clamp(neg_sim_score - pos_sim_score + margin, min=0)

        return trip_loss, trip_loss.mean()

    def sum_margin_trip_loss(self, pos_sim_score, neg_sim_score, margin):
        '''
        input:
            pos_sim_score: [n_q, n_p]
            neg_sim_score: [n_q, n_p]
            margin: float
        '''
        n_q, n_p = pos_sim_score.shape
        trip_loss = torch.clamp(neg_sim_score - pos_sim_score + margin, min=0)
        trip_loss = trip_loss.sum(dim=-1)
        
        return trip_loss, trip_loss.mean()


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
