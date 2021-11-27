import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import Bottleneck_RU, Classificiation, FrontFeatureExtractor, Refinement_block, Squeeze_Exicitation

class FishNet(nn.Module):
    def __init__(self, args):
        super(FishNet, self).__init__()
        self.n_classes = args.n_classes

        self.n_tail = args.n_tail
        self.n_body = args.n_body
        self.n_head = args.n_head
        self.in_channel = args.in_channel

        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.up_sampling = nn.Upsample(scale_factor=2)

        self._set_tail()
        self._set_body()
        self._set_head()

        self.front_fe = FrontFeatureExtractor(self.in_channel)
        self.se_block = Squeeze_Exicitation(self.in_channel*2**(self.n_tail))
        self.classifier = Classificiation(self.channels[-1], self.n_classes)

    def _set_tail(self):
        n_tail = self.n_tail

        self.channels = [self.in_channel*2**(i) for i in range(n_tail+1)] # out channels

        self.tail_module = nn.ModuleList([nn.Sequential(
            Refinement_block(self.channels[i], 2, 'tail'),
            Bottleneck_RU(self.channels[i]*2, 1),
            self.max_pool) for i in range(n_tail)])
    
    def _set_body(self):
        n_tail, n_body = self.n_tail, self.n_body

        self.tail_body_resolution = nn.ModuleList([Bottleneck_RU(c, 1) for c in self.channels[::-1]])

        for n in range(self.n_body):
            self.channels.append((self.channels[-1] + self.channels[n_tail-n])//2) 
        
        self.body_module = nn.ModuleList([nn.Sequential(
            Refinement_block(self.channels[i]*2, 0.5, 'body'), # input: concated features
            self.up_sampling) for i in range(n_tail+1, n_tail+n_body+1)])
    
    def _set_head(self):
        n_tail, n_body, n_head = self.n_tail, self.n_body, self.n_head

        self.channels.append(self.channels[-1] + self.channels[0]) # tail의 처음과 body의 마지막이 concat하여 head의 input channel이 됨
        resolution_moduls = [Bottleneck_RU(self.channels[0], 1)]

        for n in range(self.n_head-1):
            self.channels.append(self.channels[-1] + self.channels[n_tail+n_body-n]*2) # concat된 body와 이전 head feature가 더해짐
            resolution_moduls.append(Bottleneck_RU(self.channels[n_tail+n_body-n]*2, 1))

        self.channels.append(self.channels[-1] + self.channels[n_tail]) # body의 처음(tail의 마지막 channel과 같아서 공유)과 맨 마지막으로 concat
        resolution_moduls.append(Bottleneck_RU(self.channels[n_tail], 1))

        self.body_head_resolution =  nn.ModuleList(resolution_moduls)
        self.head_module = nn.ModuleList([nn.Sequential(
            Bottleneck_RU(self.channels[i], 1),
            self.max_pool) for i in range(n_tail+n_body+1, n_tail+n_body+n_head+1)])

    def forward(self, x):
        x = self.front_fe(x)
        
        # --- fishnet ---
        tail_results = [x]
        body_inputs = [x]

        for i in range(self.n_tail): # tail
            x = self.tail_module[i](x)
            tail_results.insert(0, x)

        x = self.se_block(x)
        first_body_input = x

        for i in range(self.n_body): # body
            diff_module_same_resolution_feat = self.tail_body_resolution[i](tail_results[i])
            concat_feat = torch.cat((diff_module_same_resolution_feat, x), 1)
            
            x = self.body_module[i](concat_feat)
            body_inputs.insert(1, concat_feat)
        
        for i in range(self.n_head): # head
            diff_module_same_resolution_feat = self.body_head_resolution[i](body_inputs[i])
            concat_feat = torch.cat((diff_module_same_resolution_feat, x), 1)
            x = self.head_module[i](concat_feat)
        
        # calculate predicted probability
        diff_module_same_resolution_feat = self.body_head_resolution[-1](first_body_input)
        concat_feat = torch.cat((diff_module_same_resolution_feat, x), 1)
        probs = self.classifier(concat_feat).squeeze(-1).squeeze(-1)

        return probs



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tail', type=int, default=3)
    parser.add_argument('--n_body', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--in_channel', type=int, default=64)

    args = parser.parse_args()
    
    # stage: 1 2 3 4 |(4) 3 2 1| 2 3 4
    img_sample = torch.zeros([2, 3, 32, 32])
    fishnet = FishNet(args)
    fishnet(img_sample)
