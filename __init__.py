from .nodes.comfy_nodes import UNOModelLoader, UNOGenerate


# 注册节点
NODE_CLASS_MAPPINGS = {
    "UNOModelLoader": UNOModelLoader,
    "UNOGenerate": UNOGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UNOModelLoader": "UNO Model Loader",
    "UNOGenerate": "UNO Generate",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']