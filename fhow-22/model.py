import torch.nn as nn
from timm import create_model


def get_model(model_name:str, pretrained:bool=True):
    return Model(model_name, pretrained)


class Model(nn.Module):
    def __init__(self, model_name:str, pretrained:bool):
        super().__init__()    
        self.base = create_model(model_name, pretrained=pretrained)
    
        in_feats = self.base.get_classifier().in_features
        self.linear_d = nn.Linear(in_feats, 7)
        self.linear_g = nn.Linear(in_feats, 6)
        self.linear_e = nn.Linear(in_feats, 3)

        self.base.reset_classifier(num_classes=0)
        
    def forward(self, inputs):
        outputs = self.base(inputs)
        return (
            self.linear_d(outputs),
            self.linear_g(outputs),
            self.linear_e(outputs)
        )

        
if __name__ == "__main__":
    from utils import get_config

    config = get_config()
    model = get_model(
        config.model.name,
        config.model.pretrained
    )
    print(model)
