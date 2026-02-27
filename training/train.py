import os
import argparse
import sys
import torch
import torchaudio

from dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
from install_ffmpeg import install_ffmpeg


from tqdm import tqdm
import json



# AWS SageMaker
SM_MODEL_DIR  = os.environ.get('SM_MODEL_DIR', ".")
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', "/opt/ml/data/training")
SM_CHANNEL_VALIDATION = os.environ.get('SM_CHANNEL_VALIDATION', "/opt/ml/data/validation")
SM_CHANNEL_TEST = os.environ.get('SM_CHANNEL_TEST', "/opt/ml/data/test")


 # To allow PyTorch to use more GPU memory if needed
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int , default = 20)
    parser.add_argument("--batch_size" , type = int , default = 16)
    parser.add_argument("--learning_rate",type=float , default = 0.001)

    # Data directories
    parser.add_argument("--train_dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val_dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test_dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model_dir", type=str, default=SM_MODEL_DIR)

    return parser.parse_args()



def main():
    # Install FFMPEG on AWS SageMaker instance
    if not install_ffmpeg():
        print("FFmpeg installation failed. Exiting.")
        sys.exit(1)


    print("Available Audio backends: ")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Track initial GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print(f"Initial GPU Memory Used: {memory_used:.2f} GB")


    train_loader,val_loader,test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size
    )
    

    print(f"Training CSV path: {os.path.join(args.train_dir, 'train_sent_emo.csv')}")
    print(f"Training video directory: {os.path.join(args.train_dir, 'train_splits')}")


    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader,val_loader)
    best_val_loss = float('inf')

    metrics_data = {
        'train_losses' : [],
        'val_losses' : [],
        'epochs': []
    }

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        val_loss,val_metrics=trainer.evaluate(val_loader)

        # Track metrics
        metrics_data['train_losses'].append(train_loss["total"])
        metrics_data['val_losses'].append(val_loss["total"])
        metrics_data['epochs'].append(epoch)

        # Log metrics in SageMaker format
        print(json.dumps({
            "metrics":[
                {"Name":"train:loss","Value":train_loss["total"]},
                
                {"Name":"val:loss","Value":val_loss["total"]},
                
                {"Name":"val:emotion_precision","Value":val_metrics['emotion_precision']},
                
                {"Name":"val:emotion_accuracy","Value":val_metrics['emotion_accuracy']},
                
                {"Name":"val:sentiment_precision","Value":val_metrics['sentiment_precision']},
                
                {"Name":"val:sentiment_accuracy","Value":val_metrics['sentiment_accuracy']}       
                       ]
        }))

        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(f"Peak GPU Memory Used: {memory_used:.2f} GB")

        # Save best model
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            torch.save(model.state_dict(),os.path.join(
                args.model_dir, "model.pth"
            ))

    # After training is complete , evaluate on test set
    print("Evaluating on test set...")
    test_loss,test_metrics = trainer.evaluate(test_loader,phase="test")
    metrics_data['test_loss'] = test_loss['total']

    # Log metrics in SageMaker format
    print(json.dumps({
            "metrics":[
                {"Name":"test:loss","Value":test_loss["total"]},
                
                {"Name":"test:emotion_precision","Value":test_metrics['emotion_precision']},
                
                {"Name":"test:emotion_accuracy","Value":test_metrics['emotion_accuracy']},
                
                {"Name":"test:sentiment_precision","Value":test_metrics['sentiment_precision']},
                
                {"Name":"test:sentiment_accuracy","Value":test_metrics['sentiment_accuracy']}  
                       ]
        }))



if __name__ == "__main__":
    main()