from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import torch
import subprocess
import torchaudio


import os
import cv2
import numpy as np
import warnings


warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # To suppress tokenizer parallelism warning from Hugging Face


class MELDDataset:
    def __init__(self,csv_path,video_dir):
        self.data = pd.read_csv(csv_path)
        
        self.video_dir = video_dir

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.emotion_map = {
            'anger':0,
            'disgust':1,
            'fear':2,
            'joy':3,
            'neutral':4,
            'sadness':5,
            'surprise':6
        }

        self.sentiment_map = {
            'negative':0,
            'neutral':1,
            'positive':2
        }


    def __len__(self):
        return len(self.data)
    

    def _load_video_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)
        frames=[]

        try:
            if not cap.isOpened():
                raise IOError(f"Video not found: {video_path}")

            ret,frame = cap.read()
            
            if not ret or frame is None:
                raise ValueError(f"Cannot read frames from video file {video_path}")
            
            # Reset index to not skip the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while len(frames) < 30 and cap.isOpened():
                ret,frame = cap.read()
                if not ret:
                    break

                frame=cv2.resize(frame,(224,224))
                frame = frame / 255.0 #reason to divide by 255 is to normalize the pixel values to the range [0,1]
                frames.append(frame)


        except Exception as e:
            print(f"Video error: {str(e)}")
        
        finally:
            cap.release()
        
        if(len(frames)==0):
            raise ValueError("No frames could be extracted")

        # Pad or truncate frames to ensure we have exactly 30 frames basically black bars or frames meaning empty...
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]
        
        # Before permute : [Frames , height , width , channels]
        # After permute : [Frames , channels , height , width] -> ResNet expects (C,H,W) format for each frame.
        return torch.FloatTensor(np.array(frames)).permute(0,3,1,2) # Convert to tensor and change to (C,H,W)


    def _extract_audio_features(self,video_path):
        audio_path = video_path.replace('.mp4','.wav')

        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path,
            ],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

            waveform, sample_rate = torchaudio.load(audio_path)

            # Sample rate is 16000, we can extract features like MFCCs or log-mel spectrograms here

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate,16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            mel_spec = mel_spectrogram(waveform)


            # Normalize the mel spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std())

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            
            else:
                mel_spec = mel_spec[:,:,:300] # We ensure that we have a 
                #fixed length of 300 frames for the audio features, similar 
                # to the video frames.
            
            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
            

        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


    def __getitem__(self,idx):

        if isinstance(idx,torch.Tensor):
            idx=idx.item()
        row = self.data.iloc[idx]

        try:
            # Extract video file name based on Dialogue_ID and Utterance_ID
            video_file = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
            path = os.path.join(self.video_dir,video_file)
            video_path_exists = os.path.exists(path)

            if not video_path_exists:
                raise FileNotFoundError(f"Video file {video_file} not found in {self.video_dir}")


            text_inputs = self.tokenizer(row['Utterance'], 
                                        return_tensors='pt', 
                                        padding='max_length', 
                                        truncation=True,
                                        max_length=128
                                        )
            
            video_frames= self._load_video_frames(path)
            audio_features =self._extract_audio_features(path)

            # Map emotion and sentiment labels to integers
            emotion_label = self.emotion_map[row['Emotion']]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs' : {
                    'input_ids': text_inputs['input_ids'].squeeze(), 
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing index {path}: {str(e)}")
            return None
        



def collate_fn(batch):
    # Filter out None samples 
    batch = list(filter(None,batch))
    return torch.utils.data.dataloader.default_collate(batch)


def prepare_dataloaders(train_csv,train_video_dir,
                       dev_csv,dev_video_dir,
                       test_csv,test_video_dir,batch_size=32):
    
    train_dataset = MELDDataset(train_csv,train_video_dir)
    test_dataset = MELDDataset(test_csv,test_video_dir)
    dev_dataset = MELDDataset(dev_csv,dev_video_dir)


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=batch_size,
                                collate_fn=collate_fn)
    
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 collate_fn=collate_fn)
    
    return train_dataloader,dev_dataloader,test_dataloader
    

if __name__ == "__main__":

    

    train_loader,dev_loader,test_loader = prepare_dataloaders(
        '../dataset/train/train_sent_emo.csv','../dataset/train/train_splits',
        '../dataset/dev/dev_sent_emo.csv','../dataset/dev/dev_splits_complete',
        '../dataset/test/test_sent_emo.csv','../dataset/test/output_repeated_splits_test')

    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break