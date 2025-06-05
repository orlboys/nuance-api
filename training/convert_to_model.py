import torch
import os
from datetime import datetime

def convert_existing_checkpoints_to_inference_models(checkpoints_dir):
    """
    Convert your existing checkpoints to inference-ready models
    """
    from models.bias_model import BiasModel
    
    for filename in os.listdir(checkpoints_dir):
        if filename.endswith('.pth'):
            checkpoint_path = os.path.join(checkpoints_dir, filename)
            
            try:
                # Try loading as checkpoint first
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    # It's a full checkpoint, extract model state
                    state_dict = checkpoint['model_state_dict']
                    epoch = checkpoint.get('epoch', 'unknown')
                else:
                    # It's already a state dict
                    state_dict = checkpoint
                    epoch = filename.split('_')[-1].split('.')[0]
                
                # Save as inference-ready model
                inference_path = os.path.join(checkpoints_dir, f"inference_model_epoch_{epoch}.pth")
                torch.save(state_dict, inference_path)
                print(f"✅ Converted: {filename} -> inference_model_epoch_{epoch}.pth")
                
            except Exception as e:
                print(f"❌ Error converting {filename}: {e}")


if __name__ == "__main__":
    checkpoints_dir = "./trained_models/8_epochs_2e-05_lr_512_seqlen/checkpoint"
    convert_existing_checkpoints_to_inference_models(checkpoints_dir)