from verifiers.envs import MultiTurnEnv
from typing import Dict, Any, Tuple

class KevinEnv(MultiTurnEnv):
    def __init__(self, dataset, system_prompt, max_turns=8):
        super().__init__(dataset=dataset, system_prompt=system_prompt, max_turns=max_turns)

    def initialize_state(self)

    def env_response(self, messages, state) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass

    def is_completed(self, messages, state) -> bool:
        pass

