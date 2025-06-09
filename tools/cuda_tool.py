import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

import kevin_reward as kr

class CudaTool(BaseTool):
    """A tool for calculating the reward of a generated CUDA kernel.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "calc_cuda_reward",
                "description": "A tool for calculating the reward of a generated CUDA kernel. Scaled between 0.0 and 1.0",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cuda_src": {
                            "type": "string",
                            "description": "The source code to be evaluated. Must be a valid CUDA kernel wrapped in a PyTorch harness.",
                        },
                    },
                    "required": ["cuda_src"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, original_source_code: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "original_source_code": original_source_code,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        reward = await self.calc_reward(instance_id)
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        self._instance_dict[instance_id]["reward"] = reward

        return f"LLM Generated CUDA kernel: {self._instance_dict[instance_id]['response']}", tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return kr.compute_score(
            self._instance_dict[instance_id]["original_source_code"],
            self._instance_dict[instance_id]["response"],
            perf_trials=100,
            correct_trials=5,
            weights={"compile": 0.2, "correct": 0.3, "performance": 0.5},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]