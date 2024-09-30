import logging
from abc import ABC, abstractmethod
from time import sleep
from openai import OpenAI

class LLM(ABC):
    @abstractmethod
    def __init__(self, model_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_logprobs: int | None = None,
        **kwargs
    ) -> str | list[dict]:
        raise NotImplementedError

class OpenAIChat(LLM):

    def __init__(self, model_name='gpt-3.5-turbo-0125') -> None:
        self.client = OpenAI()
        self.model_name = model_name

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_logprobs: int | None = None,
        **kwargs
    ) -> str | list[dict]:
        success = False
        failed = 0
        if top_logprobs is None:
            logprobs_flag = False
        else:
            logprobs_flag = True
        while not success:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    logprobs=logprobs_flag,
                    top_logprobs=top_logprobs,
                    **kwargs
                )
                result = response.choices[0].message.content
                if logprobs_flag:
                    log_prob_seq = response.choices[0].logprobs.content
                    logprobs = [{pos_info.token: pos_info.logprob for pos_info in position.top_logprobs} for position in log_prob_seq]
                else:
                    logprobs = dict()
                success = True
                sleep(0.5)
                return result, logprobs
            except Exception as e:
                logging.error('openai:' + str(e))
                result = 'error:{}'.format(e)
                logprobs = dict()
                failed += 1
                sleep(2.0)
            if failed > 5:
                break
        return result, logprobs

def get_llm(series: str, model_name: str) -> LLM:
    if series == "openai":
        return OpenAIChat(model_name)
    else:
        raise ValueError(f"Unknown series: {series}")