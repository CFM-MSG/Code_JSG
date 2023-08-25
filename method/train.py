import os
import sys
import time
import json
import pprint
import random
import numpy as np
import pickle
from easydict import EasyDict as EDict
from tqdm import tqdm, trange
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import h5py
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import BaseOptions
from model import GGCLNet
from data_provider import Dataset4Training,collate_train,read_video_ids
from modules.data_provider import VisDataSet4Test, TxtDataSet4Test

from eval import eval_epoch,start_inference
from modules.optimization import BertAdam
from utils.basic_utils import AverageMeter, BigFile, read_dict, log_config
from utils.model_utils import count_parameters


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, optimizer, opt, epoch_i, training=True):
    logger.info("use train_epoch func for training: {}".format(training))
    model.train(mode=training)
    if opt.hard_negative_start_epoch != -1 and epoch_i >= opt.hard_negative_start_epoch:
        model.set_hard_negative(True, opt.hard_pool_size)


    # init meters
    dataloading_time = AverageMeter()
    prepare_inputs_time = AverageMeter()
    model_forward_time = AverageMeter()
    model_backward_time = AverageMeter()
    loss_meters = OrderedDict(clip_nce_loss=AverageMeter(), clip_trip_loss=AverageMeter(),
                      frame_nce_loss=AverageMeter(), frame_trip_loss=AverageMeter(),
                              loss_overall=AverageMeter(), 
                              inter_video_trip_loss=AverageMeter(), pos_rec_loss=AverageMeter(),
                              rec_neg_query_trip_loss=AverageMeter(), rec_neg_vid_trip_loss=AverageMeter(),
                              pos_sl_rec_loss=AverageMeter(), inter_video_nce_loss=AverageMeter(),
                              intra_vid_rec_cl_loss=AverageMeter(), kd_kl_loss=AverageMeter(),
                              clip_intra_trip_loss=AverageMeter(), frame_intra_trip_loss=AverageMeter(), )
                              # frame_intra_trip_loss_hardest=AverageMeter() )

    num_training_examples = len(train_loader)

    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training Iteration", total=num_training_examples):
        global_step = epoch_i * num_training_examples + batch_idx
        dataloading_time.update(time.time() - timer_dataloading)

        # continue
        timer_start = time.time()
        for k in batch.keys():
            # if k != 'text_labels':
            batch[k] = batch[k].to(opt.device, non_blocking=opt.pin_memory) # add non_blocking=opt.pin_memory

        # model_inputs = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        prepare_inputs_time.update(time.time() - timer_start)
        timer_start = time.time()
        loss, loss_dict = model(batch['clip_video_features'],batch['frame_video_features'],batch['videos_mask'],
                                batch['text_feat'],batch['text_mask'], epoch=epoch_i) # with rec: , batch['words_id'], batch['words_feat'], batch['words_lens'], batch['words_weights']; per viddeo: ,batch['text_labels']
        model_forward_time.update(time.time() - timer_start)
        timer_start = time.time()
        if training:
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip != -1:
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            model_backward_time.update(time.time() - timer_start)

            opt.writer.add_scalar("Train/LR", float(optimizer.param_groups[0]["lr"]), global_step)
            for k, v in loss_dict.items():
                opt.writer.add_scalar("Train/{}".format(k), v, global_step)

        for k, v in loss_dict.items():
            loss_meters[k].update(float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    if training:
        to_write = opt.train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H_%M_%S"), epoch=epoch_i,
                                                      loss_str=" ".join(["{} {:.4f}".format(k, v.avg)
                                                                         for k, v in loss_meters.items()]))
        with open(opt.train_log_filepath, "a") as f:
            f.write(to_write)
        print("Epoch time stats:")
        print("dataloading_time: max {dataloading_time.max} min {dataloading_time.min} avg {dataloading_time.avg}\n"
              "prepare_inputs_time: max {prepare_inputs_time.max} "
              "min {prepare_inputs_time.min} avg {prepare_inputs_time.avg}\n"
              "model_forward_time: max {model_forward_time.max} "
              "min {model_forward_time.min} avg {model_forward_time.avg}\n"
              "model_backward_time: max {model_backward_time.max} "
              "min {model_backward_time.min} avg {model_backward_time.avg}\n".format(
            dataloading_time=dataloading_time, prepare_inputs_time=prepare_inputs_time,
            model_forward_time=model_forward_time, model_backward_time=model_backward_time))
    else:
        for k, v in loss_meters.items():
            opt.writer.add_scalar("Eval_Loss/{}".format(k), v.avg, epoch_i)


def rm_key_from_odict(odict_obj, rm_suffix):
    """remove key entry from the OrderedDict"""
    return OrderedDict([(k, v) for k, v in odict_obj.items() if rm_suffix not in k])


def train(model, train_dataset, val_video_dataset, val_text_dataset, opt):
    # Prepare optimizer
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU


    train_loader = DataLoader(dataset=train_dataset,batch_size=opt.bsz,shuffle=True,pin_memory=opt.pin_memory,
                                num_workers=opt.num_workers,collate_fn=collate_train)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    num_train_optimization_steps = len(train_loader) * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters, lr=opt.lr, weight_decay=opt.wd, warmup=opt.lr_warmup_proportion,
                         t_total=num_train_optimization_steps, schedule="warmup_linear")
    prev_best_score = 0.
    es_cnt = 0
    start_epoch = -1 if opt.eval_untrained else 0

    save_submission_filename = "latest_{}_{}_predictions_{}.json".format(opt.collection, 'val', 'VCMR')
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            # with torch.autograd.detect_anomaly():
            pass
            train_epoch(model, train_loader, optimizer, opt, epoch_i, training=True)

        with torch.no_grad():
            metrics, vcmr_sum_score = eval_epoch(model, val_video_dataset, val_text_dataset, opt, opt.results_dir, save_submission_filename, epoch=epoch_i)

        # stop_score = vcmr_sum_score # depend on task (prvr or vcmr)
        stop_metric_names = ['0.5-r1', '0.5-r10', '0.5-r100', '0.7-r1', '0.7-r10', '0.7-r100'] # 
        stop_score = sum([metrics['VCMR'][e] for e in stop_metric_names])
        if stop_score > prev_best_score:
            es_cnt = 0
            prev_best_score = stop_score
            checkpoint = {"model": model.state_dict(), "model_cfg": model.config, "epoch": epoch_i}
            torch.save(checkpoint, opt.ckpt_filepath)

            logger.info("The checkpoint file has been updated.")
        else:
            es_cnt += 1
            if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                with open(opt.train_log_filepath, "a") as f:
                    f.write("Early Stop at epoch {}".format(epoch_i))
                break
        if opt.debug:
            break

    opt.writer.close()


def start_training(opt):
   
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True


    opt.writer = SummaryWriter(opt.tensorboard_log_dir)
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Metrics] {eval_metrics_str}\n"

    rootpath = opt.root_path

    collection = opt.collection

    trainCollection = '%strain' % collection
    valCollection = '%sval' % collection

    if collection == 'activitynet' or collection == 'didemo': # not complete 
        cap_file = {'train': "prvr_vcmr_train.jsonl",
                'val': "prvr_vcmr_val.jsonl"}
    else:
        cap_file = {'train': "prvr_vcmr_train_comp.jsonl",
                    'val': "prvr_vcmr_val_comp.jsonl"}


    if collection == 'tvr':
        text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat(reloclnet-v0).hdf5' % collection)
    else:
        text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)


    # caption
    caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x])
                     for x in cap_file}
    # new visual feats
    video_feat_path = os.path.join(rootpath, collection, '%s_%s.hdf5'%(collection, opt.visual_feature))

    train_dataset = Dataset4Training(caption_files['train'], video_feat_path, text_feat_path, opt) # ,vocab

    val_text_dataset = TxtDataSet4Test(caption_files['val'], text_feat_path, opt)

    # val_video_ids_list = read_video_ids(caption_files['val'])
    val_video_dataset = VisDataSet4Test(video_feat_path, opt, caption_files['val'])


    
    model_config = EDict(
        visual_input_size=opt.visual_feat_dim,
        query_input_size=opt.q_feat_size,
        hidden_size=opt.hidden_size,  # hidden dimension
        max_ctx_l=opt.max_ctx_l,
        max_desc_l=opt.max_desc_l,
        map_size=opt.map_size,
        input_drop=opt.input_drop,
        device=opt.device_ids,
        drop=opt.drop,
        n_heads=opt.n_heads,  # self-att heads
        initializer_range=opt.initializer_range,  # for linear layer
        margin=opt.margin,  # margin for ranking loss
        use_hard_negative=False,  # reset at each epoch
        hard_pool_size=opt.hard_pool_size,
        ##############################
        pooling_ksize=opt.pooling_ksize,
        pooling_stride=opt.pooling_stride,
        conv_ksize=opt.conv_ksize,
        conv_stride=opt.conv_stride,
        num_gauss_center=opt.num_gauss_center,
        num_gauss_width=opt.num_gauss_width,
        width_lower_bound=opt.width_lower_bound,
        width_upper_bound=opt.width_upper_bound,
        window_size_inc=opt.window_size_inc,
        max_epoch=opt.n_epoch,
        sigma=opt.sigma,
        gamma=opt.gamma,
        alpha1=opt.alpha1,
        alpha2=opt.alpha2,
        alpha3=opt.alpha3,
        beta1=opt.beta1,
        beta2=opt.beta2,
        num_props=opt.num_props,
        clip_proposal=opt.clip_proposal,
        proposal_method=opt.proposal_method,
        shared_trans=opt.shared_trans,
        # intraCL_matched=opt.intraCL_matched,
        global_sample=opt.global_sample,
        intra_margin=opt.intra_margin,
        word_dim=opt.word_dim,
        vocab_size=opt.vocab_size,
        hard_negative_start_epoch=opt.hard_negative_start_epoch,
        eval_ngc=opt.eval_ngc,
        eval_ngw=opt.eval_ngw,
        props_topk=opt.props_topk,
        )
    logger.info("model_config {}".format(model_config))

    NAME_TO_MODELS = {'GGCLNet':GGCLNet}
    model = NAME_TO_MODELS[opt.model_name](model_config)
    count_parameters(model)
    
    logger.info("Start Training...")
    train(model, train_dataset, val_video_dataset, val_text_dataset, opt)
    return opt.results_dir, opt.eval_split_name, opt.eval_path, opt.debug, opt.model_name


if __name__ == '__main__':
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    log_config(opt.results_dir, 'performance')
    model_dir, eval_split_name, eval_path, debug, model_name = start_training(opt)
    if not debug:
        model_dir = model_dir.split(os.sep)[-1]

        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model in {}".format(model_dir))
        logger.info("Input args {}".format(sys.argv[1:]))
        opt.eval_id = "eval_after_train"
        opt.model_dir = opt.results_dir
        start_inference(opt)
