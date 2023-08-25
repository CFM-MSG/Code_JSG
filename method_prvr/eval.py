import numpy as np
import logging
import torch.backends.cudnn as cudnn
import os
import pickle
from method_prvr.model import GGCLNet
from torch.utils.data import DataLoader
from modules.data_provider import VisDataSet4Test,\
    TxtDataSet4Test,read_video_ids, collate_frame_val, collate_text_val
from tqdm import tqdm
from collections import defaultdict
import torch
from utils.basic_utils import AverageMeter, BigFile, read_dict, save_json, save_jsonl
from method_prvr.config import TestOptions
from standalone_eval.eval import eval_retrieval
from utils.temporal_nms import temporal_non_maximum_suppression


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def filter_vcmr_by_nms(all_video_predictions, nms_threshold=0.6, max_before_nms=1000, max_after_nms=100,
                       score_col_idx=3):
    """ Apply non-maximum suppression for all the predictions for each video.
    1) group predictions by video index
    2) apply nms individually for each video index group
    3) combine and sort the predictions
    Args:
        all_video_predictions: list(sublist),
            Each sublist is [video_idx (int), st (float), ed(float), score (float)]
            Note the scores are negative distances.
        nms_threshold: float
        max_before_nms: int
        max_after_nms: int
        score_col_idx: int
    """
    predictions_neg_by_video_group = defaultdict(list)
    for pred in all_video_predictions[:max_before_nms]:
        predictions_neg_by_video_group[pred[0]].append(pred[1:])  # [st (float), ed(float), score (float)]
    predictions_by_video_group_neg_after_nms = dict()
    for video_idx, grouped_preds in predictions_neg_by_video_group.items():
        predictions_by_video_group_neg_after_nms[video_idx] = temporal_non_maximum_suppression(
            grouped_preds, nms_threshold=nms_threshold)
    predictions_after_nms = []
    for video_idx, grouped_preds in predictions_by_video_group_neg_after_nms.items():
        for pred in grouped_preds:
            pred = [video_idx] + pred  # [video_idx (int), st (float), ed(float), score (float)]
            predictions_after_nms.append(pred)
    # ranking happens across videos, descending order
    predictions_after_nms = sorted(predictions_after_nms, key=lambda x: x[score_col_idx], reverse=True)[:max_after_nms]
    return predictions_after_nms


def post_processing_vcmr_nms(vcmr_res, nms_thd=0.6, max_before_nms=1000, max_after_nms=100):
    """
    vcmr_res: list(dict), each dict is
        {
            "desc": str,
            "desc_id": int,
            "predictions": list(sublist)  # each sublist is
                [video_idx (int), st (float), ed(float), score (float)], video_idx could be different
        }
    """
    processed_vcmr_res = []
    for e in vcmr_res:
        e["predictions"] = filter_vcmr_by_nms(e["predictions"], nms_threshold=nms_thd, max_before_nms=max_before_nms,
                                              max_after_nms=max_after_nms)
        processed_vcmr_res.append(e)
    return processed_vcmr_res

def ap_score(sorted_labels):
    nr_relevant = len([x for x in sorted_labels if x > 0])
    if nr_relevant == 0:
        return 0.0

    length = len(sorted_labels)
    ap = 0.0
    rel = 0

    for i in range(length):
        lab = sorted_labels[i]
        if lab >= 1:
            rel += 1
            ap += float(rel) / (i + 1.0)
    ap /= nr_relevant
    return ap

# mssl version
def get_gt_old(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt

# new version
# new version
def get_gt(video_metas, query_metas, cap2vid):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            # if query_id.split('#', 1)[0] == vid_id:
            if cap2vid[query_id] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt

def eval_q2m(indices, q2m_gts):
    n_q, n_m = indices.shape

    gt_ranks = np.zeros((n_q,), np.int32)
    aps = np.zeros(n_q)
    for i in range(n_q):
        sorted_idxs = indices[i]
        # sorted_idxs = np.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = np.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(np.where(gt_ranks <= 100)[0]) / n_q
    medr = np.median(gt_ranks)
    meanr = gt_ranks.mean()

    return (r1, r5, r10, r100, medr, meanr)

def eval_q2m_v2(scores, q2m_gts):
    n_q, n_m = scores.shape
    sorted_indices = np.argsort(scores)
    
    gt_list = []
    for i in sorted(q2m_gts):
        gt_list.append(q2m_gts[i][0])
    gt_list = np.array(gt_list)
    pred_ranks = np.argwhere(sorted_indices==gt_list[:, np.newaxis])[:, 1]

    r1 = 100 * (pred_ranks==0).sum() / n_q
    r5 = 100 * (pred_ranks<5).sum() / n_q
    r10 = 100 * (pred_ranks<10).sum() / n_q
    r100 = 100 * (pred_ranks<100).sum() / n_q
    medr = np.median(pred_ranks)
    meanr = pred_ranks.mean()

    return (r1, r5, r10, r100, medr, meanr)

def t2v_map(c2i, t2v_gts):
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0]*len(d_i)

        x = t2v_gts[i][0]
        labels[x] = 1

        sorted_labels = [labels[x] for x in d_i]

        current_score = ap_score(sorted_labels)
        perf_list.append(current_score)
    return np.mean(perf_list)


def compute_context_info(model, eval_dataset, opt):
    model.eval()
    n_total_vid = len(eval_dataset)
    context_dataloader = DataLoader(eval_dataset, collate_fn=collate_frame_val, batch_size=opt.eval_context_bsz,
                                    num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    bsz = opt.eval_context_bsz
    metas = []  # list(dicts)
    clip_feat, frame_feat, frame_mask = [], [], []
    clip_proposal_feat = None
    frame_props_feat = None
    for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing context info",
                           total=len(context_dataloader)):
        metas.extend(batch[-1])
        clip_video_feat_ = batch[0].to(opt.device, non_blocking=opt.pin_memory)
        frame_video_feat_ = batch[1].to(opt.device, non_blocking=opt.pin_memory)
        frame_mask_ = batch[2].to(opt.device, non_blocking=opt.pin_memory)

        _frame_feat, _clip_proposal_feat, _encoded_clip_feat, _frame_props_feat  = model.encode_context(clip_video_feat_, frame_video_feat_, frame_mask_, return_frame_props_feat=True)
        # _video_proposal_feat = _video_proposal_feat.cpu().numpy() # ori
        clip_feat.append(_encoded_clip_feat) # _frame_feat
        frame_feat.append(_frame_feat)
        frame_mask.append(frame_mask_)
        if clip_proposal_feat is None:
            clip_proposal_feat = torch.zeros((n_total_vid, _clip_proposal_feat.shape[1], opt.hidden_size)).type_as(_clip_proposal_feat)
            clip_proposal_feat[idx * bsz:(idx + 1) * bsz] = _clip_proposal_feat
        else:
            clip_proposal_feat[idx * bsz:(idx + 1) * bsz] = _clip_proposal_feat
        # frame props feat
        if  frame_props_feat is None:
            frame_props_feat = torch.zeros((n_total_vid, _frame_props_feat.shape[1], opt.hidden_size)).type_as(_frame_props_feat)
            frame_props_feat[idx * bsz:(idx + 1) * bsz] = _frame_props_feat
        else:
            frame_props_feat[idx * bsz:(idx + 1) * bsz] = _frame_props_feat
    # vid_proposal_feat = torch.from_numpy(vid_proposal_feat).to(opt.device) # ori
    def cat_tensor(tensor_list):
        if len(tensor_list) == 0:
            return None
        else:
            seq_l = [e.shape[1] for e in tensor_list]
            b_sizes = [e.shape[0] for e in tensor_list]
            b_sizes_cumsum = np.cumsum([0] + b_sizes)
            if len(tensor_list[0].shape) == 3:
                hsz = tensor_list[0].shape[2]
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
            elif len(tensor_list[0].shape) == 2:
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
            else:
                raise ValueError("Only support 2/3 dimensional tensors")
            for i, e in enumerate(tensor_list):
                res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i+1], :seq_l[i]] = e
            return res_tensor

    return dict(
        video_metas=metas,  # list(dict) (N_videos)
        clip_proposal_feat=clip_proposal_feat,
        frame_proposal_feat=frame_props_feat,
        clip_feat=cat_tensor(clip_feat), # 同一个batch内frame数相同，但不同batch不同，所以需要处理
        frame_feat = cat_tensor(frame_feat),
        video_mask=cat_tensor(frame_mask)
        )

def compute_query2ctx_info(model, eval_text_dataset, eval_video_dataset, opt, ctx_info):
    model.eval()

    query_eval_loader = DataLoader(eval_text_dataset, collate_fn=collate_text_val, batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)

    query_metas = []
    clip_scale_scores = []
    frame_scale_scores = []
    frame_atten_scores = []
    score_sum = []
    score_sum_mssl = []
    vcmr_res = [] # added
    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing query2video scores", total=len(query_eval_loader)):
        
        # print("***", idx, "***")
        _query_metas = batch[3]
        query_metas.extend(batch[3]) # changed
        query_feat = batch[0].to(opt.device) # , non_blocking=opt.pin_memory
        query_mask = batch[1].to(opt.device) # , non_blocking=opt.pin_memory

        
        # forward
        _clip_scale_scores, _frame_atten_scores, var4ml = model.get_pred_from_raw_query(
            query_feat, query_mask, ctx_info['clip_proposal_feat'], ctx_info['clip_feat'] # , ctx_info['video_mask']
        )
        num_props_for_each_video = opt.npev
        # vcmr on frame scores:
        # _q2v_props, _q2v_props_scores, _frame_scale_scores = model.ml_head.get_vcmr_inference_results(ctx_info['frame_feat'], ctx_info['video_mask'], var4ml['video_query'], num_props_for_each_video) # vcmr on frame branch
        _q2v_props, _q2v_props_scores, _frame_scale_scores = model.get_infer_res_from_props_feat(var4ml['video_query'], ctx_info['frame_proposal_feat'], num_props_for_each_video) # vcmr on frame branch
        
        # vcmr on mixed scores:
        # _q2v_props, _q2v_props_scores, _frame_scale_scores = model.get_mixed_infer_res(var4ml['video_query'], ctx_info['clip_proposal_feat'], ctx_info['frame_proposal_feat'], num_props_for_each_video) # vcmr on clip+frame branch
        
        # vcmr on clip scores:
        ## _q2v_props, _q2v_props_scores = model.get_clip_scale_ranked_props_and_scores(var4ml['video_query'], ctx_info['video_proposal_feat'], num_props_for_each_video)
        # _frame_scale_scores, _ = model.ml_head.get_moment_level_inference_results(ctx_info['frame_feat'], ctx_info['video_mask'], var4ml['video_query'])
        # _q2v_props, _q2v_props_scores, _ = model.get_infer_res_from_props_feat(var4ml['video_query'], ctx_info['clip_proposal_feat'], num_props_for_each_video)

        # vr infer
        ########################################
        
        _score_sum = opt.clip_scale_w*_clip_scale_scores + opt.frame_scale_w*_frame_scale_scores 
        _score_sum_mssl = opt.clip_scale_w*_clip_scale_scores + opt.frame_scale_w*_frame_atten_scores

        clip_scale_scores.append(_clip_scale_scores)
        frame_scale_scores.append(_frame_scale_scores)
        frame_atten_scores.append(_frame_atten_scores)
        score_sum.append(_score_sum)
        score_sum_mssl.append(_score_sum_mssl)

        # mr infer
        num_query, num_video, num_props, _ = _q2v_props.shape
        _selected_props = _q2v_props.reshape(num_query, num_video*num_props, 2)
        _q2v_topk_scores = _q2v_props_scores.reshape(num_query, num_video*num_props)


        ## sort score
        max_vcmr_props = min(opt.max_vcmr_props, num_video*num_props) # 1000
        _sorted_scores, _sorted_score_indices = torch.topk(_q2v_topk_scores, max_vcmr_props, dim=1) # ori

        _selected_props_left = _selected_props[:,:,0]
        _selected_props_right = _selected_props[:,:,1]
        _selected_props_left_sorted = torch.gather(_selected_props_left, 1, _sorted_score_indices)
        _selected_props_right_sorted = torch.gather(_selected_props_right, 1, _sorted_score_indices)
        _selected_props_sorted = torch.stack([_selected_props_left_sorted, _selected_props_right_sorted], dim=-1) # [num_query, max_vcmr_props, 2]

        _selected_props_sorted = _selected_props_sorted.cpu().numpy().copy()
        _sorted_scores = _sorted_scores.cpu().numpy().copy()
        _sorted_score_indices = _sorted_score_indices.cpu().numpy().copy()

        # process queries one by one
        for i, single_query_id in enumerate(_query_metas):
            single_query_sorted_score = _sorted_scores[i]
            desc = eval_text_dataset.captions[single_query_id]

            curr_vcmr_predictions = []

            for j, single_prop in enumerate(_selected_props_sorted[i]):
                vid_idx_local, prop_idx = np.unravel_index(_sorted_score_indices[i, j], shape=(num_video, num_props))
                vid_name = ctx_info["video_metas"][vid_idx_local] # TODO: check
                vid_idx = eval_video_dataset.vid2idx[vid_name]
                vid_duration = eval_video_dataset.vid2duration[vid_name]
                prop_duration = single_prop * vid_duration
                v_score = single_query_sorted_score[j]

                curr_vcmr_predictions.append([vid_idx, float(prop_duration[0]), 
                                                float(prop_duration[1]), float(v_score), vid_name])
            
            curr_query_pred = dict(
                desc_id = single_query_id,
                desc = desc,
                predictions = curr_vcmr_predictions
            )
            vcmr_res.append(curr_query_pred)

    clip_scale_scores = torch.cat(clip_scale_scores, dim=0)
    clip_sorted_indices = torch.argsort(clip_scale_scores, dim=1, descending=True).cpu().numpy().copy()
    frame_scale_scores = torch.cat(frame_scale_scores, dim=0)
    frame_sorted_indices = torch.argsort(frame_scale_scores, dim=1, descending=True).cpu().numpy().copy()
    frame_atten_scores = torch.cat(frame_atten_scores, dim=0)
    atten_sorted_indices = torch.argsort(frame_atten_scores, dim=1, descending=True).cpu().numpy().copy()
    score_sum = torch.cat(score_sum, dim=0)
    sum_sorted_indices = torch.argsort(score_sum, dim=1, descending=True).cpu().numpy().copy()
    score_sum_mssl = torch.cat(score_sum_mssl, dim=0)
    mssl_sum_sorted_indices = torch.argsort(score_sum_mssl, dim=1, descending=True).cpu().numpy().copy()

    vcmr_res_dict = dict(VCMR=vcmr_res)

    return clip_sorted_indices, frame_sorted_indices, atten_sorted_indices, sum_sorted_indices, mssl_sum_sorted_indices, query_metas, vcmr_res_dict


def get_perf(t2v_sorted_indices, t2v_gt):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr) = eval_q2m(t2v_sorted_indices, t2v_gt)
    t2v_map_score = t2v_map(t2v_sorted_indices, t2v_gt)


    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10_100, medr, meanr: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1), round(t2v_r100, 1)]))
    logging.info(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10+t2v_r100, 1)))
    logging.info(" * mAP: {}".format(round(t2v_map_score, 4)))
    logging.info(" * "+'-'*10)

    return (t2v_r1, t2v_r5, t2v_r10,t2v_r100, t2v_medr, t2v_meanr, t2v_map_score)


def eval_epoch(model, val_video_dataset, val_text_dataset, opt, res_dir, save_submission_filename, epoch=999):
    model.eval()
    logger.info("*"*60)
    logger.info('*'*20 + f" Eval epoch: {epoch}" + '*'*20)

    context_info = compute_context_info(model, val_video_dataset, opt)
    clip_sorted_indices, frame_sorted_indices, atten_sorted_indices, sum_sorted_indices, mssl_sum_sorted_indices, query_metas, vcmr_res_dict = compute_query2ctx_info(model,
                                                                                                        val_text_dataset,
                                                                                                        val_video_dataset,
                                                                                                        opt,
                                                                                                        context_info
                                                                                                        )
    video_metas = context_info['video_metas']

    # v2t_gt, t2v_gt = get_gt(video_metas, query_metas) # old
    v2t_gt, t2v_gt = get_gt(video_metas, query_metas, val_text_dataset.cap2vid) # new
    logger.info('clip_scale_scores:')
    clip_scale_scores = get_perf(clip_sorted_indices, t2v_gt)
    logger.info('frame_scale_scores:')
    frame_scale_scores = get_perf(frame_sorted_indices, t2v_gt)
    logger.info('attention_based_scores:') # add
    attention_based_scores = get_perf(atten_sorted_indices, t2v_gt)
    logger.info('score_sum:')
    intigrated_score = get_perf(sum_sorted_indices, t2v_gt)
    t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score = intigrated_score
    prvr_sum_score = 0
    prvr_sum_score += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)

    logger.info('score_sum_mssl:')
    get_perf(mssl_sum_sorted_indices, t2v_gt)

    ########################################
    # Event-level metrics

    vcmr_res_dict["video2idx"] = val_video_dataset.vid2idx
    
    IOU_THDS = (0.1, 0.3, 0.5, 0.7)
    submission_path = os.path.join(res_dir, save_submission_filename)
    # optical
    # save_json(vcmr_res_dict, submission_path, save_pretty=True) # pretty

    # NMS
    vcmr_res_dict_nmsed = dict(video2idx=vcmr_res_dict["video2idx"])
    nms_thd = 0.5 # opt.nms_thd (0.5)
    vcmr_res_dict_nmsed['VCMR'] = post_processing_vcmr_nms(vcmr_res_dict['VCMR'], nms_thd=nms_thd, max_before_nms=1000, max_after_nms=100) # nms_thd=0.5, # max_before_nms=1000

    # nms or not
    metrics = eval_retrieval(vcmr_res_dict_nmsed, val_text_dataset.cap_data, # vcmr_res_dict for no nms
                            iou_thds=IOU_THDS, match_number=not opt.debug, verbose=opt.debug, 
                            use_desc_type=opt.collection=='tvr') # cap_data格式和函数要求不一致，需要重新处理
    logger.info('-'*40)
    logger.info("VCMR results:")
    # logger.info(metrics)
    vcmr_sum_score = 0
    for k, v in metrics['VCMR'].items():
        vcmr_sum_score += v
        logger.info(str(k) + ': ' + str(v))
    logger.info(f"vcmr_sum_score: {vcmr_sum_score}")
    logger.info('*'*40 + '\n')
    metrics['vcmr_sum_score'] = vcmr_sum_score
    # metrics['prvr_frame_scale_score'] = 
    metrics['prvr_attention_based_scores'] = attention_based_scores
    metrics['prvr_frame_scale_scores'] = frame_scale_scores
    metrics['prvr_clip_scale_score'] = clip_scale_scores
    metrics['prvr_intigrated_score'] = intigrated_score
    metrics['prvr_sum_score'] = prvr_sum_score
    save_metrics_path = submission_path.replace(".json", "_nms_metrics.json") # nms or not
    save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)

    return metrics, vcmr_sum_score


def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    ckpt_filepath = os.path.join(opt.ckpt_filepath)
    checkpoint = torch.load(ckpt_filepath, map_location=torch.device('cuda:%d' % opt.device_ids[0])) # changed
    loaded_model_cfg = checkpoint["model_cfg"]
    NAME_TO_MODELS = {'GGCLNet':GGCLNet}
    model = NAME_TO_MODELS[opt.model_name](loaded_model_cfg)
    
    model.load_state_dict(checkpoint["model"], strict=False) # changed
    logger.info("Loaded model saved at epoch {} from checkpoint: {}".format(checkpoint["epoch"], opt.ckpt_filepath))

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU
    return model

def start_inference(opt=None):
    logger.info("Setup config, data and model...")
    if opt is None:
        opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    rootpath = opt.root_path
    collection = opt.collection
    testCollection = '%stest' % collection

    # cap_file = {'test': '%s.caption.txt' % testCollection}
    if collection == 'activitynet' or collection == 'didemo': # not complete 
        cap_file = {'test': "prvr_vcmr_test.jsonl"}
    else:
        cap_file = {'test': "prvr_vcmr_test_comp.jsonl"}

    # caption
    caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x])
                     for x in cap_file}

    if collection == 'tvr':
        text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat(reloclnet-v0).hdf5' % collection)
    else:
        text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
    
    # new visual feats
    video_feat_path = os.path.join(rootpath, collection, '%s_%s.hdf5'%(collection, opt.visual_feature))

    # test_video_ids_list = read_video_ids(caption_files['test'])
    test_vid_dataset = VisDataSet4Test(video_feat_path, opt, caption_files['test'])
    test_text_dataset = TxtDataSet4Test(caption_files['test'], text_feat_path, opt)



    model = setup_model(opt)

    save_submission_filename = "inference_{}_{}_{}.json".format('val', opt.eval_id, 'VCMR')

    logger.info("Starting inference...")
    with torch.no_grad():
        prvr_score, vcmr_score = eval_epoch(model, test_vid_dataset, test_text_dataset, opt, opt.model_dir, save_submission_filename)



if __name__ == '__main__':
    start_inference()