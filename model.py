import torch.nn as nn
from torchvision import models

class FruitQualityModel(nn.Module):
    def __init__(self, num_fruits, num_qualities):
        super(FruitQualityModel, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Identity()  # Remove the classifier
        
        # Define two separate classifiers
        self.fruit_classifier = nn.Linear(num_ftrs, num_fruits)
        self.quality_classifier = nn.Linear(num_ftrs, num_qualities)
    
    def forward(self, x):
        features = self.base_model(x)
        fruit_output = self.fruit_classifier(features)
        quality_output = self.quality_classifier(features)
        return fruit_output, quality_output
