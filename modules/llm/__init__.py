from .chat import NODE_CLASS_MAPPINGS as CHAT_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CHAT_NODE_DISPLAY_NAME_MAPPINGS
from .crynux import NODE_CLASS_MAPPINGS as CRYNUX_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CRYNUX_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(CHAT_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(CRYNUX_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(CHAT_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CRYNUX_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
