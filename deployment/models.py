import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models


import warnings



warnings.filterwarnings("ignore")

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased') #Use the base BERT model for text encoding on top of the pre-trained weights


        for param in self.bert.parameters():
            # Freeze BERT parameters to prevent them from being updated during training
            param.requires_grad = False 

        self.projection = nn.Linear(768, 128) # Project BERT's 768-dimensional output to a smaller 128-dimensional space
        

    def forward(self, input_ids, attention_mask):
        #Extract BERT embeddings for the input text
        outputs =  self.bert(input_ids = input_ids, attention_mask = attention_mask)


        # Use the [CLS] token representation
        pooler_output = outputs.pooler_output 

        return self.projection(pooler_output)
    

class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        # Use a pre-trained 3D ResNet-18 model for video encoding
        self.backbone = vision_models.video.r3d_18(pretrained=True) 

        for param in self.backbone.parameters():
            # Freeze the backbone parameters to prevent them from being updated during training
            param.requires_grad = False


        # Get the number of features from the backbone's final fully connected layer
        num_fts = self.backbone.fc.in_features 
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128), # Project the backbone's output to a smaller
            nn.ReLU(), # Add a non-linear activation function
            nn.Dropout(0.2) # Add dropout for regularization
        )

    
    def forward(self,x):
        # [batch_size, frames,channels, height, width] -> [batch_size,channels,frames, height, width]
        x = x.transpose(1,2)
        return self.backbone(x)
    




class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64,128,kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

      # Note:

        # 1. The "Flatten" Approach (Older Models)
        #     If you use Flatten(), you are keeping the spatial location of features. 
        #     The model learns that "a cat's ear is usually in the top-left."


        # 2. The "AdaptiveAvgPool" Approach (Modern Models)
        #     If you use AdaptiveAvgPool2d(1), you are telling the model to 
        #     find the "average intensity" of each feature across the whole image, 
        #     regardless of where it is.

        for param in self.conv_layers.parameters():
            # Freeze the convolutional layers to prevent them from being updated during training
            param.requires_grad = False
        

        self.projection = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self,x):
        # [batch_size,64,300] -> [batch_size,128,1]
        x=x.squeeze(1)
        features = self.conv_layers(x)
        # Features output is [batch_size,128,1] -> [batch_size,128] as linear 
        # layer expects (batch_size, features) input

        return self.projection(features.squeeze(-1))
    


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super(MultimodalSentimentModel,self).__init__()

        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()


        # Fusion layer to combine text, video, and audio features
        self.fusion_layer = nn.Sequential(
            nn.Linear(128*3, 256), # Combine the three modalities (128 each) into a single 256-dimensional vector
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3) 
        )

        # Classification layer

        self.emo_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7) # Output layer for 7 emotion classes(Sadness,happy,etc..)
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3) # Output layer for 3 sentiment classes
        )

    def forward(self,text_inputs,video_frames,audio_features):
        text_features = self.text_encoder(
            text_inputs['input_ids'], 
            text_inputs['attention_mask'])
        
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate features from all three modalities
        combined_features = torch.cat(
            [text_features, video_features, audio_features], 
        dim=1) #[Batch size, 128*3]

        # Pass through the fusion layer
        fused_features = self.fusion_layer(combined_features) 

        # Get emotion and sentiment predictions
        emotion_output = self.emo_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output
        }