from functools import partial
import torch
import torch.nn as nn

# Search
class Search(nn.Module):
    def __init__(self):
        super(Search, self).__init__()

    def forward(self, weight_list_v, weight_list_a):
        select_weight_list = []
        for ww, ww2 in zip(weight_list_v, weight_list_a):
            select_weight_list.append(torch.matmul(ww, ww2))
        tmp = torch.stack(select_weight_list, -1)
        new_select_weight, _ = tmp.max(-1)
        last_map = new_select_weight[:, :, 0, 1:]
        # _:
        _, max_inx = last_map.max(2)
        return _, max_inx

class Select(nn.Module):
    def __init__(self, bs, device, embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_ratio, attn_drop_ratio):
        super(Select, self).__init__()
        self.time_video = nn.Parameter(torch.zeros(1, embed_dim))
        self.time_audio = nn.Parameter(torch.zeros(1, embed_dim))

        self.time_drop_video = nn.Dropout(p=drop_ratio)
        self.time_drop_audio = nn.Dropout(p=drop_ratio)

        self.BN_video = nn.BatchNorm1d(embed_dim)
        self.BN_audio = nn.BatchNorm1d(embed_dim)

        self.cls_video_w = torch.nn.Parameter(torch.ones(1)).to(device)
        self.cls_audio_w = torch.nn.Parameter(torch.ones(1)).to(device)

        # self.FF = nn.Linear(embed_dim * bs * 2, embed_dim)
        self.embed_dim = embed_dim

        self.pos_drop_video = nn.Dropout(p=drop_ratio)
        self.pos_drop_audio = nn.Dropout(p=drop_ratio)
        self.bs = bs
        self.LN = partial(nn.LayerNorm, eps=1e-6)
        self.device = device

    def forward(self, cls_video, cls_audio):
        cls_video_n = cls_video.reshape(cls_video.shape[0], 1, 768)
        cls_audio_n = cls_audio.reshape(cls_audio.shape[0], 1, 768)

        cls_video_n = cls_video_n.reshape(cls_video.shape[0], 768)
        cls_audio_n = cls_audio_n.reshape(cls_video.shape[0], 768)
        if cls_video.shape[0] > 1:
            cls_video_n = self.BN_video(cls_video_n)
            cls_audio_n = self.BN_audio(cls_audio_n)

        cls_video_n = cls_video_n.flatten()
        cls_audio_n = cls_audio_n.flatten()  # [6144]

        fusion_cls = torch.cat((self.cls_video_w * cls_video_n, self.cls_audio_w * cls_audio_n), -1)
        FF = nn.Linear(self.embed_dim * cls_video.shape[0] * 2, self.embed_dim).to(self.device)
        fusion_cls = FF(fusion_cls)
        res = torch.sigmoid(fusion_cls)
        pre_num = 0

        ys = []
        for i in range(1, self.embed_dim):
            if self.embed_dim % i == 0:
                ys.append(i)

        for i in res:
            if i >= 0.7:  # changable
                pre_num += 1
        # too many
        if pre_num > self.embed_dim/2:
            num = 24   # Fixed maximum value
        else:
            if self.embed_dim % pre_num == 0:
                num = pre_num
            else:
                ys.append(pre_num)
                ys.sort()
                num = ys[ys.index(pre_num) + 1]
        return num