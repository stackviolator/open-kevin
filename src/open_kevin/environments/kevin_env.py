from verifiers.envs import MultiTurnEnv
from typing import Dict, Any, Tuple
from copy import deepcopy
from openai import OpenAI
from typing import Union, List
from verifiers.parsers import XMLParser
from open_kevin.rewards import compute_score_modular

class KevinEnv(MultiTurnEnv):
    def __init__(self, dataset, system_prompt, max_turns: int = 8):
        # Use the shared XMLParser from verifiers for <think> and <code> tags
        self.parser = XMLParser(fields=["think", "code"])
        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            parser=self.parser,
            max_turns=max_turns,
        )

    def initialize_state(self):
        return {
            "messages": [],
            "last_summary": "",
            "runtime_stats": "",
            "compiler_errors": "",
            "attempts": 0,
            "score": 1.0,
        }

    def env_response(self, messages, state, *, client: OpenAI, model: str, sampling_args=None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compress the assistant's verbose CoT by generating an internal summary and
        overwriting the last assistant message so it contains the summary and the
        unchanged <code> block. Returns a metadata system message.
        """

        sampling_args = sampling_args or {}

        # Work on a copy of state so original isn't mutated unexpectedly
        state = deepcopy(state)
        state["attempts"] += 1

        assistant_text = messages[-1]["content"]
        # Parse <think> and <code>
        try:
            parsed = self.parser.parse(assistant_text)
            cot_text = getattr(parsed, "think", "") or ""
            kernel_body = getattr(parsed, "code", "") or ""
        except Exception:
            cot_text, kernel_body = "", assistant_text

        # 1. Create summary via internal LLM call
        if cot_text:
            summary_prompt = [
                {"role": "system", "content": "Summarize the following chain-of-thought into at most two sentences."},
                {"role": "user", "content": cot_text},
            ]
            summary_text = self.get_model_response(
                prompt=summary_prompt,
                client=client,
                model=model,
                sampling_args={"temperature": 0.6, "max_tokens": 200},
                message_type="chat",
            )
        else:
            summary_text = ""

        compact_content = (
            f"<think>\n{summary_text}\n</think>\n<code>\n{kernel_body}\n</code>"
        )

        # Overwrite the last assistant message with compact version
        messages[-1]["content"] = compact_content

        # 2. Evaluate kernel once (cached)
        from open_kevin.rewards.base import _get_kernel_result
        ref_prompt = state.get("ref_prompt", "")
        answer = state.get("answer", "")
        kb_res = _get_kernel_result(ref_prompt, answer, compact_content)
        state["compiler_errors"] = kb_res.metadata.get("error", "") if not kb_res.compiled else ""
        state["runtime_stats"] = str(getattr(kb_res, "runtime", ""))
        state["score"] = compute_score_modular(ref_prompt, compact_content, answer)
        state["last_summary"] = summary_text

        # 3. Build metadata system message
        env_msg = {
            "role": "user",
            "content": (
                "[METADATA]\n"
                f"attempt: {state['attempts']} / {self.max_turns}\n"
                f"runtime_stats: {state['runtime_stats']}\n"
                f"compiler_errors: {state['compiler_errors']}\n"
                f"score: {state['score']}\n"
            ),
        }

        return env_msg, state

    def is_completed(self, messages, state) -> bool:
        print(f"state: {state}")
        print(f"messages: {messages}")
        if state["attempts"] >= self.max_turns:
            return True
        return False
    
    def rollout(self,
                client: OpenAI,
                model: str,
                prompt: Union[str, List[Dict[str, Any]]],
                answer: str,
                task: str = "default",
                info: Dict[str, Any] = {},
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        is_completed = False
        state = self.initialize_state()
        assert isinstance(prompt, list)
        messages = deepcopy(prompt) 
        completion = []
        turn = 0
        while not is_completed:
            if self.is_completed(messages, state, **kwargs):
                is_completed = True
                break
            response = self.get_model_response(
                prompt=messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type
            )
            has_error = response.startswith("[ERROR]")
            messages.append({"role": "assistant", "content": response})
            completion.append({"role": "assistant", "content": response})
            turn += 1
            if self.is_completed(messages, state, **kwargs) or turn >= self.max_turns or has_error:
                is_completed = True
            else:
                env_msg, state = self.env_response(
                    messages,
                    state,
                    client=client,
                    model=model,
                    sampling_args=sampling_args,
                    **kwargs,
                )
                messages.append(env_msg)
                completion.append(env_msg)
        return completion, state