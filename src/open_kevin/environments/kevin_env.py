from verifiers.envs import MultiTurnEnv
from typing import Dict, Any, Tuple
from copy import deepcopy
from openai import OpenAI
from typing import Union, List
from verifiers.parsers import XMLParser
from open_kevin.rewards import compute_score_modular
from phoenix.trace import suppress_tracing

class KevinEnv(MultiTurnEnv):
    def __init__(self, dataset, system_prompt, max_turns: int = 8, parser: XMLParser = None, **kwargs):
        # Use the shared XMLParser from verifiers for <think> and <code> tags
        self.parser = parser
        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            parser=self.parser,
            max_turns=max_turns,
            **kwargs,
        )

    def initialize_state(self):
        return {
            "messages": [],
            "last_summary": "",
            "runtime_stats": "",
            "compiler_errors": "",
            "attempts": 0,
            "score": 0.0,
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

        # Evaluate kernel once (cached)
        from open_kevin.rewards.base import _get_kernel_result
        kb_res = _get_kernel_result(prompt=messages[1]["content"], completion=messages[-1]["content"])
        if not kb_res.compiled:
            state["compiler_errors"] = kb_res.metadata.get("compilation_error") if kb_res.metadata.get("compilation_error") else "Compilation failed."
        else:
            state["compiler_errors"] = ""
        state["runtime_stats"] = str(getattr(kb_res, "runtime", ""))
        state["correctness"] = kb_res.correctness
        state["score"] = compute_score_modular(prompt=messages[-2]["content"], completion=messages[-1]["content"])

        # Create summary via internal LLM call
        if cot_text not in "":
            summary_prompt = [
                {"role": "system", "content": "Summarize the following chain-of-thought to capture just the main ideas of what happened."},
                {"role": "user", "content": cot_text},
            ]
            with suppress_tracing():
                summary_text = self.get_model_response(
                    prompt=summary_prompt,
                    client=client,
                    model=model,
                    sampling_args={"temperature": 0.6, "max_tokens": 500},
                    message_type="chat",
                )

            # Overwrite the last assistant message with compact version
            messages[-1]["content"] = f"<think>\n{summary_text}\n</think>\n<code>\n{kernel_body if kernel_body != '' else 'No code or incorrectly formatted code provided.'}\n</code>"


        # 3. Build metadata system message
        guidance = (
            "[NEXT_ACTION]\n"
            "If 'compiler_errors' is not empty, fix the errors and produce a new kernel. "
            "If the kernel compiles but is incorrect, fix the errors and produce a new kernel. "
            "If the kernel is correct but slow, keep it correct and try to make it faster. "
            "Return your reasoning in <think> tags followed by the updated <code> block."
        )

        env_msg = {
            "role": "user",
            "content": (
                "Here is metadata about the previously generated kernel:\n"
                "[METADATA]\n"
                f"compiled: {kb_res.compiled}\n"
                f"correct: {state['correctness']}\n"
                f"runtime_stats: {state['runtime_stats']}\n"
                f"compiler_errors: {state['compiler_errors']}\n"
                f"score: {state['score']}\n\n" + guidance
            ),
        }

        return env_msg, state

    def is_completed(self, messages, state) -> bool:
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
            messages.append({"role": "assistant", "content": response})
            turn += 1
            if self.is_completed(messages, state, **kwargs):
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
        return messages, state