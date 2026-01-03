import torch
import torch.nn as nn
import open_clip
import math
from model.Necker import Necker
from model.Adapter import Adapter
from model.Classifier import Classifier
from model.CoOp import PromptMaker  

class SCFAD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading CLIP model: {config.model_name}...")
        
        self.clip_model, preprocess, self.model_cfg = open_clip.create_model_and_transforms(config.model_name, config.image_size, device=self.device)
        
        for param in self.clip_model.parameters():
            param.requires_grad_(False)
            
        self._setup_token_info()

        self.necker = Necker(clip_model=self.clip_model).to(self.device)
        
        self.adapter = Adapter(
            clip_model=self.clip_model, 
            target=self.model_cfg['embed_dim']
        ).to(self.device)
        
        self.adapter_diff = Adapter(
            clip_model=self.clip_model, 
            target=1024
        ).to(self.device)
        
        self.Classifier = Classifier(image_size=config.image_size).to(self.device)

        if config.prompt_maker == 'coop':
            self.prompt_maker = PromptMaker(
                prompts=config.prompts,
                clip_model=self.clip_model,
                n_ctx=config.n_learnable_token,
                CSC=config.CSC,
                class_token_position=config.class_token_positions,
            ).to(self.device)
        else:
            raise NotImplementedError("Prompt maker type must be ['coop']")

    def _setup_token_info(self):
        with torch.no_grad():
            img = torch.ones((1, 3, self.config.image_size, self.config.image_size)).to(self.device)
            
            _, tokens = self.clip_model.encode_image(img, self.config.layers_out)

            if len(tokens[0].shape) == 3:
                self.clip_model.token_size = [int(math.sqrt(token.shape[1]-1)) for token in tokens]
                self.clip_model.token_c = [token.shape[-1] for token in tokens]
            else:
                self.clip_model.token_size = [token.shape[2] for token in tokens]
                self.clip_model.token_c = [token.shape[1] for token in tokens]

            self.clip_model.embed_dim = self.model_cfg['embed_dim']
            print(f"Model token size: {self.clip_model.token_size}, Token dim: {self.clip_model.token_c}")

    def encode_features(self, images):
        with torch.no_grad():
            _, image_tokens = self.clip_model.encode_image(images, out_layers=self.config.layers_out)
            image_features = self.necker(image_tokens)
        return image_features

    def forward(self, images, images_normal):

        image_features = self.encode_features(images)
        image_normal_features = self.encode_features(images_normal)


        vision_adapter_features = self.adapter(image_features)

        vision_adapter_features_normal = [i.permute(0, 2, 3, 1) for i in image_normal_features]
        
        vision_adapter_diff_features = self.adapter_diff(image_features)

        prompt_adapter_features = self.prompt_maker(image_features)

        anomaly_map = self.Classifier(
            vision_adapter_features,
            prompt_adapter_features,
            vision_adapter_features_normal,
            vision_adapter_diff_features
        )

        return anomaly_map
    

if __name__ == "__main__":
    from easydict import EasyDict
    import yaml
    config_path = "/mnt/extra_data/cyw/project/cywCLIP/config/mvtec_Nm_Re_Cp/mvtec_Nm0.3_Re0.2_Cp0.5/bottle.yaml"
    
    with open(config_path) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    model = SCFAD(config)

    x1 = torch.randn((2, 3, config.image_size, config.image_size)).to(model.device)
    x2 = torch.randn((2, 3, config.image_size, config.image_size)).to(model.device)
    anomaly_map = model(x1, x2)
    print("Anomaly map shape:", anomaly_map.shape)
    # print(model)