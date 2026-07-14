import torch
from mmengine.config import Config
from mmseg.models import build_segmentor
from mmengine.registry import init_default_scope

init_default_scope('mmseg')
cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus_efficientnetv2s-10k_plantsegwheat.py')

# Build model
model = build_segmentor(cfg.model)
model.eval()

# Test forward
dummy_input = torch.randn(1, 3, 512, 512)
# Create proper batch_img_metas
batch_img_metas = [{
    'img_shape': (512, 512),
    'ori_shape': (512, 512),
    'pad_shape': (512, 512),
    'scale_factor': (1.0, 1.0)
}]

with torch.no_grad():
    # Extract features
    features = model.backbone(dummy_input)
    print("Backbone outputs:")
    for i, f in enumerate(features):
        print(f"  Feature {i}: {f.shape}")
    
    # Test decode head forward
    seg_logits = model.decode_head.forward(features)
    print(f"\nDecode head output: {seg_logits.shape}")
    print(f"Expected: torch.Size([1, 9, 128, 128]) or similar")
    
    # Get predictions
    predictions = seg_logits.argmax(dim=1)
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"Unique predicted classes: {torch.unique(predictions)}")
    print(f"Should be in range [0, 8] for 9 classes")
