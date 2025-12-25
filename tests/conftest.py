import sys
from unittest.mock import MagicMock

# Mock the fifo_tool_airlock_model_env modules before importing
sys.modules['fifo_tool_airlock_model_env'] = MagicMock()
sys.modules['fifo_tool_airlock_model_env.common'] = MagicMock()
sys.modules['fifo_tool_airlock_model_env.common.models'] = MagicMock()
sys.modules['fifo_tool_airlock_model_env.sdk'] = MagicMock()
sys.modules['fifo_tool_airlock_model_env.sdk.client_sdk'] = MagicMock()
