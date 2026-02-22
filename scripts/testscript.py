import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from biometric_ml.models.fusion import BiometricFusionModel

def test_model():
    print("Creating model...")
    model = BiometricFusionModel(num_classes=45)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: trainable={trainable:,} / total={total:,}")
    
    model.eval()
    with torch.no_grad():
        # Test with different input ranges to find what works
        test_cases = [
            ("Random [0,1]", torch.rand(4, 1, 224, 224)),
            ("Random [0,255]", torch.rand(4, 1, 224, 224) * 255),
            ("Zeros", torch.zeros(4, 1, 224, 224)),
            ("Ones", torch.ones(4, 1, 224, 224)),
        ]
        
        for name, fp_data in test_cases:
            batch = {
                "fingerprint": fp_data,
                "iris_left": torch.randn(4, 1, 64, 64),
                "iris_right": torch.randn(4, 1, 64, 64),
            }
            
            out = model(batch)
            fp_feat = out  # We can't easily extract intermediate, but check final diversity
            
            preds = out.argmax(dim=-1)
            unique_preds = preds.unique().numel()
            
            print(f"\n{name}: preds={preds.tolist()}, unique={unique_preds}")
            print(f"  Output stats: mean={out.mean():.3f}, std={out.std():.3f}")
            
            if unique_preds > 1:
                print(f"  ✅ PASS: Diverse predictions")
            else:
                print(f"  ❌ FAIL: Collapsed to single class")

if __name__ == "__main__":
    test_model()