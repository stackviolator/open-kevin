from verifiers.envs import MultiTurnEnv
from typing import Dict, Any, Tuple

class KevinEnv(MultiTurnEnv):
    def __init__(self, dataset, system_prompt, max_turns=8):
        super().__init__(dataset=dataset, system_prompt=system_prompt, max_turns=max_turns)

    def initialize_state(self, task_data):
        return {
            "last_summary": "",
            "runtime_stats": "",
            "compiler_errors": "",
            "attempts": 0,
            "score": 1.0
        }

    def env_response(self, messages, state) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
       Mimic Kevin paper

       1. Get last response and metadata
       2. Summarize last thought process
       3. Create new version of state with summary and metadata
       4. Increment attempts

       Note, state gets completely replaced with whats returned :D 
       Note, will prob need to redo the rollout() logic to handle completion dicts
       """

        pass

    def is_completed(self, messages, state) -> bool:
        print(f"state: {state}")
        print(f"messages: {messages}")
        if state["attempts"] >= self.max_turns:
            return True
        return False