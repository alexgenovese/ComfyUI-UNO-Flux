import os
import sys
import struct
import torch
from safetensors.torch import load_file as load_sft

# Add the UNO path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'uno'))

def analyze_ae_file(file_path):
    """Analyze the AutoEncoder file to understand its format"""
    print(f"Analyzing file: {file_path}")
    
    if not os.path.exists(file_path):
        print("File does not exist!")
        return
    
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    
    # Read first 64 bytes to analyze header
    with open(file_path, 'rb') as f:
        header = f.read(64)
        print(f"First 16 bytes (hex): {header[:16].hex()}")
        print(f"First 16 bytes (ascii): {header[:16]}")
        
        # Check for safetensors format
        f.seek(0)
        try:
            header_len = struct.unpack('<Q', f.read(8))[0]
            print(f"Potential safetensors header length: {header_len}")
            if 0 < header_len < 100000:
                header_data = f.read(header_len)
                print(f"Header data preview: {header_data[:100]}")
        except:
            print("Not a safetensors format")
    
    # Try different loading methods
    print("\n--- Testing loading methods ---")
    
    # Method 1: safetensors
    try:
        print("Testing safetensors loading...")
        sd = load_sft(file_path, device="cpu")
        print(f"✅ Safetensors loading successful! Keys: {len(sd)}")
        print(f"Sample keys: {list(sd.keys())[:5]}")
        return sd
    except Exception as e:
        print(f"❌ Safetensors loading failed: {e}")
    
    # Method 2: torch.load with weights_only=False
    try:
        print("Testing torch.load (weights_only=False)...")
        sd = torch.load(file_path, map_location="cpu", weights_only=False)
        print(f"✅ Torch.load (weights_only=False) successful! Type: {type(sd)}")
        if isinstance(sd, dict):
            print(f"Keys: {len(sd)}, Sample keys: {list(sd.keys())[:5]}")
        return sd
    except Exception as e:
        print(f"❌ Torch.load (weights_only=False) failed: {e}")
    
    # Method 3: torch.load with weights_only=True
    try:
        print("Testing torch.load (weights_only=True)...")
        sd = torch.load(file_path, map_location="cpu", weights_only=True)
        print(f"✅ Torch.load (weights_only=True) successful! Type: {type(sd)}")
        if isinstance(sd, dict):
            print(f"Keys: {len(sd)}, Sample keys: {list(sd.keys())[:5]}")
        return sd
    except Exception as e:
        print(f"❌ Torch.load (weights_only=True) failed: {e}")
    
    print("❌ All loading methods failed!")
    return None

if __name__ == "__main__":
    # Test with your specific file
    ae_file = "/comfyui/models/vae/ae.sft"  # Update this path
    
    if len(sys.argv) > 1:
        ae_file = sys.argv[1]
    
    analyze_ae_file(ae_file)
