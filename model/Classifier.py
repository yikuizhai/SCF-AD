
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

similarity_loss = torch.nn.CosineSimilarity()

class Classifier(nn.Module):

    def __init__(self, image_size):
        super(Classifier, self).__init__()
        self.image_size = image_size
        self.conv = nn.Sequential(nn.Conv2d(1024, 2, kernel_size=1))
        self.classficier = nn.Conv2d(6, 2, kernel_size=1)

    def forward(self, vision_adapter_features, propmt_adapter_features, vision_adapter_features_normal,vision_adapter_diff_features):
        vis_text = []  

        for i, vision_adapter_feature in enumerate(vision_adapter_features):
            B, H, W, C = vision_adapter_feature.shape
            anomaly_map = (vision_adapter_feature.view((B, H * W, C)) @ propmt_adapter_features).contiguous().view(
                (B, H, W, -1)).permute(0, 3, 1, 2)

            vis_text.append(anomaly_map)
        
        diff = []
        for i in range(len(vision_adapter_features_normal)):
            fs = vision_adapter_features_normal[i].permute(0,3,1,2)
            ft = vision_adapter_diff_features[i].permute(0,3,1,2)

            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            a_map = 1 - similarity_loss(fs_norm, ft_norm)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=224, mode='bilinear', align_corners=False)
            diff.append(a_map)
        diff = torch.cat(diff,dim=1)

        vis_text = torch.stack(vis_text, dim=0).mean(dim=0)
        vis_text = F.interpolate(vis_text, (self.image_size, self.image_size), mode='bilinear',
                                    align_corners=True)



        vis_text = torch.softmax(vis_text, dim=1)
        diff = torch.softmax(diff, dim=1)

        anomaly_map = torch.cat([vis_text,diff], dim=1)
        anomaly_map = self.classficier(anomaly_map)
        return torch.softmax(anomaly_map, dim=1)