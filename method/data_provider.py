import json
import torch
import torch.utils.data as data
import numpy as np
import re
import h5py
import nltk
from utils.basic_utils import load_jsonl


def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def average_to_fixed_length(visual_input, map_size):
    visual_input = torch.from_numpy(visual_input)
    num_sample_clips = map_size
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0).numpy()


    return new_visual_input

def uniform_feature_sampling_ori(features, max_len): # ori version
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = []
    for i in range(max_len):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx], axis=0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features
# pytorch version(may be faster)
def uniform_feature_sampling(visual_input, map_size):
    num_clips = visual_input.shape[0]
    if map_size is None or num_clips <= map_size:
        return visual_input
    visual_input = torch.from_numpy(visual_input)
    num_sample_clips = map_size
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0).numpy()


    return new_visual_input



def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)



def collate_train(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    # clip_video_features, frame_video_features, captions, idxs, cap_ids, video_ids, words_ids_, words_feats_, weightss_, words_lens_ = zip(*data) # with rec
    clip_video_features, frame_video_features, cap_feat, idxs, cap_id, video_id = zip(*data)

    #videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    #captions
    feat_dim = cap_feat[0].shape[-1]

    roberta_length = [len(e) for e in cap_feat]

    padded_roberta_feat = torch.zeros(len(cap_feat), max(roberta_length), feat_dim)
    roberta_mask = torch.zeros(len(cap_feat), max(roberta_length))

    for index, cap in enumerate(cap_feat):
        end = roberta_length[index]
        padded_roberta_feat[index, :end, :] = cap[:end, :]
        roberta_mask[index, :end] = 1.0

    return dict(clip_video_features=clip_videos,
                frame_video_features=frame_videos,
                videos_mask=videos_mask,
                text_feat=padded_roberta_feat,
                text_mask=roberta_mask,
                )


class Dataset4Training(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, video_feat_path, text_feat_path, opt, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.cap2vid = {}
        # self.video2frames = video2frames

        self.cap_data = load_jsonl(cap_file)
        self.data_ratio = opt.train_data_ratio
        if self.data_ratio != 1:
            n_examples = int(len(self.cap_data) * self.data_ratio)
            self.cap_data = self.cap_data[:n_examples]
            print("Using {}% of the data for training: {} examples".format(self.data_ratio * 100, n_examples))
        
        for idx, item in enumerate(self.cap_data):
            # vid_id, duration, st_ed, cap_id, caption = item # v1
            vid_id = item['vid_name']
            # query_id = item['desc_id']
            # st_ed = item['ts']
            cap_id = item['desc_id']
            caption = item['desc']
            self.captions[cap_id] = caption
            self.cap_ids.append(cap_id)
            self.cap2vid[cap_id] = vid_id
            if vid_id not in self.video_ids:
                self.video_ids.append(vid_id)
            if vid_id in self.vid_caps:
                self.vid_caps[vid_id].append(cap_id)
            else:
                self.vid_caps[vid_id] = []
                self.vid_caps[vid_id].append(cap_id)
            

        # self.visual_feat = visual_feat
        self.text_feat_path = text_feat_path
        # new vid feat
        self.visual_feat = h5py.File(video_feat_path, 'r')

        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        self.max_desc_len = opt.max_desc_l

        # self.open_file = False
        # self.length = len(self.vid_caps) # per video
        self.length = len(self.cap_data) # per query

        self.text_feat = h5py.File(self.text_feat_path, 'r')

        # add
        self.frame_feat_sample = opt.frame_feat_sample


    def __getitem__(self, index):

        cap_id = self.cap_ids[index]
        video_id = self.cap2vid[cap_id]


        # new vid feat:
        frame_vecs = self.visual_feat[video_id][...]

        clip_video_feature = average_to_fixed_length(frame_vecs, self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)

        if self.frame_feat_sample == 'fixed':
            frame_video_feature = average_to_fixed_length(frame_vecs, self.max_ctx_len)
        else:
            frame_video_feature = uniform_feature_sampling(frame_vecs, self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        # text
        cap_feat = self.text_feat[str(cap_id)][...]
        cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]


        return clip_video_feature, frame_video_feature, cap_tensor, index, cap_id, video_id # no rec

    def __len__(self):
        return self.length


if __name__ == '__main__':
    pass


