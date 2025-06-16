import os
import torch
import numpy as np
import re
from PIL import Image
from typing import Literal
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from comfy.model_management import get_torch_device
import folder_paths

# from uno.flux.modules.conditioner import HFEmbedder
# from uno.flux.pipeline import UNOPipeline, preprocess_ref
# from uno.flux.util import configs, print_load_warning, set_lora
from safetensors.torch import load_file as load_sft

# Prova import con gestione errori
try:
    from uno.flux.modules.conditioner import HFEmbedder
    from uno.flux.pipeline import UNOPipeline, preprocess_ref
    from uno.flux.util import configs, print_load_warning, set_lora
    print("✅ UNO modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing UNO modules: {e}")
    # Verifica se esistono i file
    uno_path = os.path.join(os.path.dirname(__file__), '..', 'uno')
    print(f"UNO directory exists: {os.path.exists(uno_path)}")
    if os.path.exists(uno_path):
        print(f"UNO directory contents: {os.listdir(uno_path)}")
    raise e


# 添加自定义加载模型的函数
def custom_load_flux_model(model_path, device, use_fp8, lora_rank=512, lora_path=None):
    """
    从指定路径加载 Flux 模型
    """
    from uno.flux.model import Flux
    from uno.flux.util import load_model
    
    if use_fp8:
        params = configs["flux-dev-fp8"].params
    else:
        params = configs["flux-dev"].params
    
    # 初始化模型
    with torch.device("meta" if model_path is not None else device):
        model = Flux(params)
    
    # 如果有lora，设置 LoRA 层
    if lora_path and os.path.exists(lora_path):
        print(f"Using only_lora mode with rank: {lora_rank}")
        model = set_lora(model, lora_rank, device="meta" if model_path is not None else device)
    
    # 加载模型权重
    if model_path is not None:
        print(f"Loading Flux model from {model_path}")
        
        # Only load LoRA if path exists
        lora_sd = {}
        if lora_path and os.path.exists(lora_path):
            print("Loading lora")
            try:
                if lora_path.endswith("safetensors"):
                    lora_sd = load_sft(lora_path, device=str(device))
                else:
                    lora_sd = torch.load(lora_path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Warning: Failed to load LoRA from {lora_path}: {e}")
                lora_sd = {}
        
        print("Loading main checkpoint")
        if model_path.endswith('safetensors'):
            if use_fp8:
                print(
                    "####\n"
                    "We are in fp8 mode right now, since the fp8 checkpoint of XLabs-AI/flux-dev-fp8 seems broken\n"
                    "we convert the fp8 checkpoint on flight from bf16 checkpoint\n"
                    "If your storage is constrained"
                    "you can save the fp8 checkpoint and replace the bf16 checkpoint by yourself\n"
                )
                sd = load_sft(model_path, device="cpu")
                sd = {k: v.to(dtype=torch.float8_e4m3fn, device=device) for k, v in sd.items()}
            else:
                sd = load_sft(model_path, device=str(device))
            
            sd.update(lora_sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        else:
            dit_state = torch.load(model_path, map_location='cpu', weights_only=False)
            sd = {}
            for k in dit_state.keys():
                sd[k.replace('module.','')] = dit_state[k]
            sd.update(lora_sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            model.to(str(device))
        print_load_warning(missing, unexpected)

    return model

def custom_load_ae(ae_path, device):
    """
    从指定路径加载自编码器
    """
    from uno.flux.modules.autoencoder import AutoEncoder
    from uno.flux.util import load_model
    
    # 获取对应模型类型的自编码器参数
    ae_params = configs["flux-dev"].ae_params
    
    # 初始化自编码器
    with torch.device("meta" if ae_path is not None else device):
        ae = AutoEncoder(ae_params)
    
    # 加载自编码器权重
    if ae_path is not None:
        print(f"Loading AutoEncoder from {ae_path}")
        
        try:
            # First check if file exists and is readable
            if not os.path.exists(ae_path):
                raise FileNotFoundError(f"AutoEncoder file not found: {ae_path}")
            
            # Check file size to ensure it's not corrupted/empty
            file_size = os.path.getsize(ae_path)
            if file_size < 1024:  # Less than 1KB is likely corrupted
                raise ValueError(f"AutoEncoder file appears to be corrupted (size: {file_size} bytes): {ae_path}")
            
            print(f"AutoEncoder file size: {file_size / (1024*1024):.1f} MB")
            
            # Try to use our safe loading method first
            try:
                from uno.flux.util import load_ae_safe
                print("Using safe AutoEncoder loading method...")
                sd = load_ae_safe(ae_path, device=str(device))
            except (ImportError, AttributeError):
                print("Safe loading method not available, using fallback methods...")
                
                # Determine file format and loading strategy
                file_ext = os.path.splitext(ae_path)[1].lower()
                print(f"File extension: {file_ext}")
                
                # Try loading strategies in order of preference
                loading_strategies = []
                
                if file_ext in ['.safetensors', '.sft']:
                    # .sft files are often safetensors with different extension
                    loading_strategies.append(("safetensors", lambda: load_sft(ae_path, device=str(device))))
                
                # Try to bypass ComfyUI's torch.load wrapper
                def bypass_torch_load():
                    import pickle
                    with open(ae_path, 'rb') as f:
                        return pickle.load(f)
                
                loading_strategies.extend([
                    ("direct_pickle", bypass_torch_load),
                    ("torch_load_direct", lambda: torch.load(ae_path, map_location=str(device), weights_only=False)),
                ])
                
                # Try each loading strategy
                last_error = None
                for strategy_name, load_func in loading_strategies:
                    try:
                        print(f"Attempting to load using strategy: {strategy_name}")
                        sd = load_func()
                        print(f"Successfully loaded using strategy: {strategy_name}")
                        break
                    except Exception as e:
                        print(f"Failed with strategy {strategy_name}: {e}")
                        last_error = e
                        continue
                else:
                    # All strategies failed
                    print(f"All AutoEncoder loading strategies failed. Last error: {last_error}")
                    raise last_error
            
            # Validate the loaded state dict
            if not isinstance(sd, dict):
                raise ValueError(f"Loaded state dict is not a dictionary, got {type(sd)}")
            
            if len(sd) == 0:
                raise ValueError("Loaded state dict is empty")
            
            print(f"Successfully loaded state dict with {len(sd)} keys")
            
            # Print some key names for debugging
            sample_keys = list(sd.keys())[:5]
            print(f"Sample keys: {sample_keys}")
            
            missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
            if len(missing) > 0:
                print(f"Missing keys: {len(missing)}")
                if len(missing) <= 10:  # Only print if not too many
                    print(f"Missing keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected keys: {len(unexpected)}")
                if len(unexpected) <= 10:  # Only print if not too many
                    print(f"Unexpected keys: {unexpected}")
            
            # 转移到目标设备
            ae = ae.to(str(device))
            print("AutoEncoder loaded and moved to device successfully")
            
        except Exception as e:
            print(f"Critical error loading AutoEncoder from {ae_path}: {e}")
            print("This might indicate a corrupted or incompatible AutoEncoder file.")
            print("Suggestions:")
            print("1. Try running the debug script: python debug_ae_loading.py")
            print("2. Check if the file is actually in safetensors format")
            print("3. Try renaming .sft to .safetensors if it's actually safetensors")
            print("4. Consider re-downloading the AutoEncoder model")
            raise e
            
    return ae

def custom_load_t5(model_path: str, device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    try:
        cache_dir = folder_paths.get_folder_paths("clip")[0]
        print(f"Loading T5 model with max_length={max_length}")
        
        # Ensure we're using the correct model type for T5
        model_name = os.path.basename(model_path).lower()
        if "t5" in model_name:
            # This is definitely a T5 model, use T5 configuration
            embedder = HFEmbedder(model_path, max_length=max_length, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
            # Verify the model is actually T5
            if hasattr(embedder.hf_module, 'config') and hasattr(embedder.hf_module.config, 'model_type'):
                if embedder.hf_module.config.model_type != 't5':
                    print(f"Warning: Expected T5 model but got {embedder.hf_module.config.model_type}")
            return embedder.to(device)
        else:
            print(f"Warning: Model name {model_name} doesn't contain 't5', may not be a T5 model")
            return HFEmbedder(model_path, max_length=max_length, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device)
    except Exception as e:
        print(f"Error loading T5 model from {model_path}: {e}")
        # Try without cache_dir as fallback
        try:
            return HFEmbedder(model_path, max_length=max_length, torch_dtype=torch.bfloat16).to(device)
        except Exception as e2:
            print(f"Fallback T5 loading also failed: {e2}")
            raise e2

def custom_load_clip(model_path: str, device: str | torch.device = "cuda") -> HFEmbedder:
    try:
        cache_dir = folder_paths.get_folder_paths("clip")[0]
        print(f"Loading CLIP model with max_length=77")
        
        # Ensure we're using the correct model type for CLIP
        model_name = os.path.basename(model_path).lower()
        if "clip" in model_name:
            # This is definitely a CLIP model, use CLIP configuration
            embedder = HFEmbedder(model_path, max_length=77, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
            # Verify the model is actually CLIP
            if hasattr(embedder.hf_module, 'config') and hasattr(embedder.hf_module.config, 'model_type'):
                if embedder.hf_module.config.model_type not in ['clip', 'clip_text_model']:
                    print(f"Warning: Expected CLIP model but got {embedder.hf_module.config.model_type}")
            return embedder.to(device)
        else:
            print(f"Warning: Model name {model_name} doesn't contain 'clip', may not be a CLIP model")
            return HFEmbedder(model_path, max_length=77, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device)
    except Exception as e:
        print(f"Error loading CLIP model from {model_path}: {e}")
        # Try different approaches as fallbacks
        try:
            # Fallback 1: Try without cache_dir
            print("Trying CLIP fallback 1: without cache_dir")
            return HFEmbedder(model_path, max_length=77, torch_dtype=torch.bfloat16).to(device)
        except Exception as e2:
            try:
                # Fallback 2: Try with float16 instead of bfloat16
                print("Trying CLIP fallback 2: with float16")
                cache_dir = folder_paths.get_folder_paths("clip")[0]
                return HFEmbedder(model_path, max_length=77, torch_dtype=torch.float16, cache_dir=cache_dir).to(device)
            except Exception as e3:
                try:
                    # Fallback 3: Try with float16 and no cache_dir
                    print("Trying CLIP fallback 3: float16 without cache_dir")
                    return HFEmbedder(model_path, max_length=77, torch_dtype=torch.float16).to(device)
                except Exception as e4:
                    print(f"All CLIP loading fallbacks failed. Original error: {e}")
                    print(f"Fallback 1 error: {e2}")
                    print(f"Fallback 2 error: {e3}")
                    print(f"Fallback 3 error: {e4}")
                    raise e4



class UNOModelLoader:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "UNO_MODEL"
        self.loaded_model = None

    @classmethod
    def INPUT_TYPES(cls):
        # 获取 unet 模型列表和 vae 模型列表
        model_paths = folder_paths.get_filename_list("unet")
        vae_paths = folder_paths.get_filename_list("vae")
        
        # 增加 LoRA 模型选项
        lora_paths = folder_paths.get_filename_list("loras")
        
        # 增加 CLIP 模型选项 (T5 and CLIP models)
        clip_paths = folder_paths.get_filename_list("clip")
        
        return {
            "required": {
                "flux_model": (model_paths, ),
                "ae_model": (vae_paths, ),
                "t5_model": (clip_paths, ),
                "clip_model": (clip_paths, ),
                "use_fp8": ("BOOLEAN", {"default": False}),
                "offload": ("BOOLEAN", {"default": False}),
                "lora_model": (["None"] + lora_paths, ),
            }
        }

    RETURN_TYPES = ("UNO_MODEL",)
    RETURN_NAMES = ("uno_model",)
    FUNCTION = "load_model"
    CATEGORY = "UNO"

    def load_model(self, flux_model, ae_model, t5_model, clip_model, use_fp8, offload, lora_model=None):
        device = get_torch_device()
        
        try:
            # 获取模型文件的完整路径
            flux_model_path = folder_paths.get_full_path("unet", flux_model)
            ae_model_path = folder_paths.get_full_path("vae", ae_model)
            t5_model_path = folder_paths.get_full_path("clip", t5_model)
            clip_model_path = folder_paths.get_full_path("clip", clip_model)
            
            # Validate model paths exist
            for path, name in [(flux_model_path, "Flux"), (ae_model_path, "AE"), 
                              (t5_model_path, "T5"), (clip_model_path, "CLIP")]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{name} model file not found: {path}")
            
            # 获取LoRA模型路径（如果有）
            lora_model_path = None
            if lora_model is not None and lora_model != "None":
                lora_model_path = folder_paths.get_full_path("loras", lora_model)
                if not os.path.exists(lora_model_path):
                    print(f"Warning: LoRA model file not found: {lora_model_path}")
                    lora_model_path = None
            
            print(f"Loading Flux model from: {flux_model_path}")
            print(f"Loading AE model from: {ae_model_path}")
            print(f"Loading T5 model from: {t5_model_path}")
            print(f"Loading CLIP model from: {clip_model_path}")
            lora_rank = 512
            if lora_model_path:
                print(f"Loading LoRA model from: {lora_model_path}")
            
            # 创建自定义 UNO Pipeline
            class CustomUNOPipeline(UNOPipeline):
                def __init__(self, use_fp8, device, flux_path, ae_path, t5_path, clip_path, offload=False, 
                            lora_rank=512, lora_path=None):
                    self.device = device
                    self.offload = offload
                    self.model_type = "flux-dev-fp8" if use_fp8 else "flux-dev"
                    self.use_fp8 = use_fp8
                    
                    try:
                        # Validate model paths and types before loading
                        t5_name = os.path.basename(t5_path).lower()
                        clip_name = os.path.basename(clip_path).lower()
                        
                        print(f"T5 model filename: {t5_name}")
                        print(f"CLIP model filename: {clip_name}")
                        
                        # Check if models are swapped
                        if "clip" in t5_name and "t5" in clip_name:
                            print("Warning: It appears T5 and CLIP models may be swapped!")
                            print("Swapping model paths...")
                            t5_path, clip_path = clip_path, t5_path
                        elif "clip" in t5_name:
                            print("Warning: T5 path contains a CLIP model. This may cause issues!")
                        elif "t5" in clip_name:
                            print("Warning: CLIP path contains a T5 model. This may cause issues!")
                        
                        # 加载 CLIP 和 T5 编码器
                        print("Loading CLIP model...")
                        self.clip = custom_load_clip(clip_path, device="cpu" if offload else self.device)
                        print("CLIP model loaded successfully")
                        
                        print("Loading T5 model...")
                        self.t5 = custom_load_t5(t5_path, device="cpu" if offload else self.device, max_length=512)
                        print("T5 model loaded successfully")
                        
                        # Validate model configurations
                        if hasattr(self.clip, 'max_length') and self.clip.max_length != 77:
                            print(f"Warning: CLIP model max_length is {self.clip.max_length}, expected 77")
                        
                        if hasattr(self.t5, 'max_length') and self.t5.max_length != 512:
                            print(f"Warning: T5 model max_length is {self.t5.max_length}, expected 512")
                        
                        print("Loading AutoEncoder...")
                        # 加载自定义模型
                        self.ae = custom_load_ae(ae_path, device="cpu" if offload else self.device)
                        print("AutoEncoder loaded successfully")
                        
                        print("Loading Flux model...")
                        self.model = custom_load_flux_model(
                            flux_path, 
                            device="cpu" if offload else self.device, 
                            use_fp8=use_fp8,
                            lora_rank=lora_rank,
                            lora_path=lora_path
                        )
                        print("Flux model loaded successfully")
                        
                    except Exception as e:
                        print(f"Error in CustomUNOPipeline initialization: {e}")
                        import traceback
                        traceback.print_exc()
                        raise e
                    
            # 创建自定义 pipeline
            model = CustomUNOPipeline(
                use_fp8=use_fp8,
                device=device,
                flux_path=flux_model_path,
                ae_path=ae_model_path,
                t5_path=t5_model_path,
                clip_path=clip_model_path,
                offload=offload,
                lora_rank=lora_rank,
                lora_path=lora_model_path,
            )
            
            self.loaded_model = model
            print(f"UNO model loaded successfully with custom models.")
            return (model,)
        except Exception as e:
            print(f"Error loading UNO model: {e}")
            print(f"Model paths attempted:")
            print(f"  Flux: {flux_model_path if 'flux_model_path' in locals() else 'Not set'}")
            print(f"  AE: {ae_model_path if 'ae_model_path' in locals() else 'Not set'}")
            print(f"  T5: {t5_model_path if 't5_model_path' in locals() else 'Not set'}")
            print(f"  CLIP: {clip_model_path if 'clip_model_path' in locals() else 'Not set'}")
            if lora_model_path:
                print(f"  LoRA: {lora_model_path}")
            import traceback
            traceback.print_exc()
            raise e


class UNOGenerate:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uno_model": ("UNO_MODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 3407}),
                "pe": (["d", "h", "w", "o"], {"default": "d"}),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "UNO"

    def generate(self, uno_model, prompt, width, height, guidance, num_steps, seed, pe, 
                reference_image_1=None, reference_image_2=None, reference_image_3=None, reference_image_4=None):
        # Make sure width and height are multiples of 16
        width = (width // 16) * 16
        height = (height // 16) * 16
        
        # Process reference images if provided
        ref_imgs = []
        ref_tensors = [reference_image_1, reference_image_2, reference_image_3, reference_image_4]
        for ref_tensor in ref_tensors:
            if ref_tensor is not None:
                # Convert from tensor to PIL
                if isinstance(ref_tensor, torch.Tensor):
                    # Handle batch of images
                    if ref_tensor.dim() == 4:  # [batch, height, width, channels]
                        for i in range(ref_tensor.shape[0]):
                            img = ref_tensor[i].cpu().numpy()
                            ref_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                            # Determine reference size based on number of reference images
                            ref_size = 512 if len([t for t in ref_tensors if t is not None]) <= 1 else 320
                            ref_image_pil = preprocess_ref(ref_image_pil, ref_size)
                            ref_imgs.append(ref_image_pil)
                    else:  # [height, width, channels]
                        img = ref_tensor.cpu().numpy()
                        ref_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                        # Determine reference size based on number of reference images
                        ref_size = 512 if len([t for t in ref_tensors if t is not None]) <= 1 else 320
                        ref_image_pil = preprocess_ref(ref_image_pil, ref_size)
                        ref_imgs.append(ref_image_pil)
                elif isinstance(ref_tensor, np.ndarray):
                    # Assume ComfyUI range is [-1, 1], convert to [0, 1]
                    ref_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                    # Determine reference size based on number of reference images
                    ref_size = 512 if len([t for t in ref_tensors if t is not None]) <= 1 else 320
                    ref_image_pil = preprocess_ref(ref_image_pil, ref_size)
                    ref_imgs.append(ref_image_pil)
        
        try:
            # Generate image
            output_img = uno_model(
                prompt=prompt,
                width=width,
                height=height,
                guidance=guidance,
                num_steps=num_steps,
                seed=seed,
                ref_imgs=ref_imgs,
                pe=pe
            )
            
            # Save the generated image
            output_filename = f"uno_{seed}_{prompt[:20].replace(' ', '_')}.png"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Convert to ComfyUI-compatible tensor
            if hasattr(output_img, 'images') and len(output_img.images) > 0:
                # Handle FluxPipelineOutput
                output_img.images[0].save(output_path)
                print(f"Saved UNO generated image to {output_path}")
                image = np.array(output_img.images[0]) / 255.0  # Convert to [0, 1]
            else:
                # Handle PIL Image
                output_img.save(output_path)
                print(f"Saved UNO generated image to {output_path}")
                image = np.array(output_img) / 255.0  # Convert to [0, 1]
            
            # Convert numpy array to torch.Tensor
            image = torch.from_numpy(image).float()
            
            # Make sure it's in ComfyUI format [batch, height, width, channels]
            if image.dim() == 3:  # [height, width, channels]
                image = image.unsqueeze(0)  # Add batch dimension to make it [1, height, width, channels]
            
            
            return (image,)
        except Exception as e:
            print(f"Error generating image with UNO: {e}")
            raise e


# Register our nodes to be used in ComfyUI
NODE_CLASS_MAPPINGS = {
    "UNOModelLoader": UNOModelLoader,
    "UNOGenerate": UNOGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UNOModelLoader": "UNO Model Loader",
    "UNOGenerate": "UNO Generate",
}
