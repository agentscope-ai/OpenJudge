import asyncio
import os
from typing import Any, Dict, List, Optional, Generator, ClassVar, Union
import pickle
import datetime
from loguru import logger
import requests
import logging
import time

from openai import OpenAI
from pydantic import Field, BaseModel, model_validator
from rm_gallery.core.model.message import ChatMessage, ChatResponse, GeneratorChatResponse, MessageRole


def get_from_dict_or_env(
    data: Dict[str, Any],
    key: str,
    default: Optional[str] = None,
) -> str:
    """Get a value from a dictionary or an environment variable.

    Args:
        data: The dictionary to look up the key in.
        key: The key to look up in the dictionary or environment. This can be a list of keys to try
            in order.
        default: The default value to return if the key is not in the dictionary
            or the environment. Defaults to None.
    """
    if key in data and data[key]:
        return data[key]
    elif key.upper() in os.environ and os.environ[key.upper()]:
        return os.environ[key.upper()]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{key.upper()}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )


def _convert_chat_message_to_openai_message(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    try:
        return [
            {
                "role": message.role.name.lower(),
                "content": message.content or "",
            } for message in messages
        ]
    except:
        try:
            return [
                {
                    "role": str(message.role).lower(),
                    "content": message.content or "",
                } for message in messages
            ]
        except:
            return [
                {
                    "role": str(message["role"]).lower(),
                    "content": str(message["content"]) or "",
                } for message in messages
            ]

def _convert_openai_response_to_response(response: Any) -> ChatResponse:
    message = response.choices[0].message
    additional_kwargs = {"token_usage": getattr(response, "usage", {})}

    message = ChatMessage(
        role=getattr(message, "role", "assistant"),
        content=getattr(message, "content", ""),
        name=getattr(message, "name", None),
        tool_calls=getattr(message, "tool_calls", None),
        additional_kwargs=additional_kwargs
    )

    return ChatResponse(
        message=message,
        raw=response.model_dump() if hasattr(response, "model_dump") else vars(response),
        additional_kwargs=additional_kwargs
    )


def _convert_stream_chunk_to_response(chunk: Any) -> Optional[ChatResponse]:
    if not chunk.choices:
        return None

    delta = chunk.choices[0].delta
    if not delta.content and not hasattr(delta, "role"):
        return None

    message = ChatMessage(
        role="assistant",
        content=delta.content or "",
        name=getattr(delta, "name", None),
        tool_calls=getattr(delta, "tool_calls", None),
        additional_kwargs={}
    )

    return ChatResponse(
        message=message,
        raw=chunk.model_dump() if hasattr(chunk, "model_dump") else vars(chunk),
        delta=message,
        additional_kwargs={"token_usage": getattr(chunk, "usage", {})}
    )


class BaseLLM(BaseModel):
    model: str
    temperature: float = 0.85
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_tokens: int = Field(default=2048, description="Max tokens to generate for llm.")
    stop: List[str] = Field(default_factory=list, description="List of stop words")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of tools to use")
    tool_choice: Union[str, Dict] = Field(default="auto", description="tool choice when user passed the tool list")
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    retry_delay: int = Field(default=60, description="Delay in seconds between retries")

    @staticmethod
    def _convert_messages(messages: List[ChatMessage] | ChatMessage | str) -> List[ChatMessage]:
        if isinstance(messages, list):
            return messages
        elif isinstance(messages, str):
            return [ChatMessage(content=messages, role=MessageRole.USER)]
        elif isinstance(messages, ChatMessage):
            assert messages.role == MessageRole.USER, "Only support user message."
            return [messages]
        else:
            raise ValueError(
                f"Invalid message type {messages}. "
            )

    def chat(self, messages: List[ChatMessage] | str, **kwargs) -> ChatResponse | GeneratorChatResponse:
        """

        Args:
            messages:
            **kwargs:

        Returns:

        """

        raise NotImplementedError

    def register_tools(self, tools: List[Dict[str, Any]], tool_choice: Union[str, Dict]):
        self.tools = tools
        self.tool_choice = tool_choice

    def chat_batched(self, messages_batched: List[List[ChatMessage]] | str, **kwargs) -> List[ChatResponse]:
        """

        Args:
            messages_batched: List of List of ChatMessage
            **kwargs: same with `chat`

        Returns:

        """
        try:
            return asyncio.get_event_loop().run_until_complete(self._chat_batched(messages_batched, **kwargs))
        except RuntimeError as e:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop().run_until_complete(self._chat_batched(messages_batched, **kwargs))

    async def _chat_batched(self, messages_batched: List[List[ChatMessage]] | str, **kwargs) -> List[ChatResponse]:
        """
        Used by `chat_batched`, do not call this method directly.
        """
        responses = await asyncio.gather(
            *[
                self.achat(msg, **kwargs) for msg in messages_batched
            ]
        )
        return responses

    async def achat(self, messages: List[ChatMessage] | str, **kwargs) -> ChatResponse | GeneratorChatResponse:
        """

        Args:
            messages:
            **kwargs:

        Returns:

        """
        result = await asyncio.to_thread(self.chat, messages, **kwargs)
        return result

    def simple_chat(self, query: str, history: Optional[List[str]] = None, sys_prompt: str = "", debug: bool = False) -> Any:
        if debug:
            # Save input arguments to cache
            cache_data = {
                'query': query,
                'history': history,
                'sys_prompt': sys_prompt
            }
            date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs(f'log/llamakit', exist_ok=True)
            with open(f'log/llamakit/cache-{date_time}.pkl', 'wb') as f:
                pickle.dump(cache_data, f)

        messages = [ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt)]

        if history is None:
            history_ = []
        else:
            history_ = history.copy()
        history_ += [query]

        for i, h in enumerate(history_):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            messages += [ChatMessage(role=role, content=h)]

        # Implement retry logic with max_retries
        for attempt in range(self.max_retries):
            try:
                response: ChatResponse = self.chat(messages)
                return response.message.content
            except Exception as e:
                if attempt < self.max_retries - 1:  # Don't sleep on the last attempt
                    logger.warning(f"Attempt {attempt + 1} failed. Retrying in {self.retry_delay} seconds...")
                    import time
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed.")
                    raise e # Re-raise the last exception if all retries failed


class LLMClient(BaseLLM):
    llm: Any
    model: str = Field(default="gpt-4o")
    econml_api_key: Optional[str] = Field(default=None)
    boyin_api_key: Optional[str] = Field(default=None)
    base_url: str = Field(default="http://8.130.177.212:3000/v1")
    boyin_base_url: str = Field(default="https://poc-dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")
    max_retries: int = Field(default=10)
    stream: bool = Field(default=False)

    # Available models list
    AVAILABLE_MODELS: ClassVar[List[str]] = [
        "o1-preview",
        "o1-mini",
        "chatgpt-4o-latest",
        "gpt-4o",
        "gpt-4o-mini",
        "claude-3-5-sonnet-latest",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-thinking-exp-1219",
        "qwen-max",
        "qwen-plus",
        "qwen-max-2025-01-25",
        "deepseek-reasoner",
        "deepseek-chat",
        "deepseek-r1",
        "deepseek-sen",
        "model-all",
        "model-data",
        "qwen2.5-72b-instruct",
        "pai-judge",
        "pai-judge",
        "qwq-plus",
        "qwq-plus-latest",
        "qwq-plus-2025-03-05",
        "qwq-32b",
        "qwq-32b-preview",
        "boyin-chat",
        "pre-boyin_chat",
        "boyin_plus",
        "deepseek-v3-0324",
        "qwen3-235b-a22b",
        "qwen3-32b",
        "qwen3-30b-a3b",
        "qwen3-14b"
    ]

    @classmethod
    def _is_boyin_model_type(cls, model: str) -> bool:
        """Check if the model is a boyin type model.
        
        Args:
            model: The model name to check
            
        Returns:
            bool: True if the model is a boyin type, False otherwise
        """
        boyin_model_identifiers = [
            "boyin",
            "cong"
            # Add new identifiers here in the future
        ]
        return any(identifier in model.lower() for identifier in boyin_model_identifiers)

    @model_validator(mode="before")
    @classmethod
    def validate_client(cls, data: Dict):
        """Create an OpenAI client for Blt."""
        # Check for ECONML_API_KEY
        econml_api_key = get_from_dict_or_env(data=data, key="ECONML_API_KEY", default=None)
        if not econml_api_key:
            raise ValueError("ECONML_API_KEY environment variable is not set. Please set it before using the client.")
        data["econml_api_key"] = econml_api_key

        # Check for BOYIN_API_KEY if using boyin model
        model = data.get("model", "gpt-4o")
        if cls._is_boyin_model_type(model):
            boyin_api_key = get_from_dict_or_env(data=data, key="BOYIN_API_KEY", default=None)
            if not boyin_api_key:
                raise ValueError("BOYIN_API_KEY environment variable is required when using boyin models. Please set it before using the client.")
            data["boyin_api_key"] = boyin_api_key
        else:
            data["boyin_api_key"] = get_from_dict_or_env(data=data, key="BOYIN_API_KEY", default="")


        try:
            data["client"] = OpenAI(
                api_key=data["econml_api_key"],
                base_url=data.get("base_url", "http://8.130.177.212:3000/v1"),
                max_retries=data.get("max_retries", 10),
                timeout=300.0
            )
            return data
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

    def _is_boyin_model(self) -> bool:
        """Check if the current model is a boyin model."""
        return self._is_boyin_model_type(self.model)

    @property
    def chat_kwargs(self) -> Dict[str, Any]:
        call_params = {
            "model": self.model,
            # "top_p": self.top_p,
            # "temperature": self.temperature,
            # "max_tokens": self.max_tokens,
            # "stream": self.stream,
        }

        # Remove None values
        call_params = {k: v for k, v in call_params.items()
                       if v is not None and (isinstance(v, bool) or v != 0)}

        if self.tools:
            call_params.update({
                "tools": self.tools,
                "tool_choice": self.tool_choice
            })

        return call_params

    def chat(self, messages: List[ChatMessage] | str, **kwargs) -> ChatResponse | Generator:
        messages = self._convert_messages(messages)

        # 如果是 boyin 模型，使用专用的 API 调用
        if self._is_boyin_model():
            return self._call_boyin_api(messages, **kwargs)
        
        # 原有的 API 调用逻辑
        call_params = self.chat_kwargs.copy()
        call_params.update(kwargs)

        try:
            response = self.client.chat.completions.create(
                messages=_convert_chat_message_to_openai_message(messages),
                **call_params
            )

            if self.stream:
                return self._handle_stream_response(response)
            return _convert_openai_response_to_response(response)

        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

    def send_request(self, messages: List[Dict], retry_count: int = 3) -> Dict:
        """发送API请求并处理重试逻辑"""
        
        url = "https://poc-dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {
            "Authorization":os.getenv("BOYIN_API_KEY"),
            "Content-Type": "application/json",
            "X-DashScope-DataInspection": "enable"  # 添加这个头部
        }
        data = {
            "model": self.model,
            "input": {
                "messages": messages
            },
            "param": {
                "enable_search": False
            }
        } 
        for attempt in range(retry_count):
            try:
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()  # Raises an HTTPError for bad responses
                return response.json()
            except Exception as e:
                if attempt == retry_count - 1:
                    logging.error(f"Request failed after {retry_count} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        return None
    
    def _call_boyin_api(self, messages: List[ChatMessage], **kwargs) -> ChatResponse | Generator:
        """Call the boyin API endpoint."""
        formatted_messages = _convert_chat_message_to_openai_message(messages)
        
        # 使用新的send_request方法
        response_data = self.send_request(formatted_messages)
        
        if not response_data:
            raise Exception("Boyin API call failed: No response data returned")
            
        return self._convert_boyin_response(response_data)
    
    def _convert_boyin_response(self, response_data: Dict) -> ChatResponse:
        """Convert boyin API response to ChatResponse format."""
        content = response_data.get("output", {}).get("text", "")
        
        message = ChatMessage(
            role="assistant",
            content=content,
            additional_kwargs={}
        )
        
        return ChatResponse(
            message=message,
            raw=response_data,
            additional_kwargs={}
        )

    def _handle_stream_response(self, response: Any) -> Generator[ChatResponse, None, None]:
        _response = None
        for chunk in response:
            chunk_response = _convert_stream_chunk_to_response(chunk)
            if chunk_response is None:
                continue

            if _response is None:
                _response = chunk_response
            else:
                _response.message = _response.message + chunk_response.message
                _response.delta = chunk_response.message

            yield _response

    def simple_chat(self, query: str, history: Optional[List[str]] = None, sys_prompt: str = "You are a helpful assistant.", debug: bool = False) -> Any:
        """Simple interface for chat with history support."""

        if "qwq" in self.model or "deepseek" in self.model or "qwen3" in self.model:
            return self.simple_chat_reasoning(query=query, history=history, sys_prompt=sys_prompt, debug=debug)

        if debug:
            # Save input arguments to cache
            cache_data = {
                'query': query,
                'history': history,
                'sys_prompt': sys_prompt
            }
            date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs(f'log/llamakit', exist_ok=True)
            with open(f'log/llamakit/cache-{date_time}.pkl', 'wb') as f:
                pickle.dump(cache_data, f)
                
        if "boyin" in self.model or "cong" in self.model:
            messages = []
        else:
            messages = [{"role": "system", "content": sys_prompt}]

        if history is None:
            history_ = []
        else:
            history_ = history.copy()
        history_ += [query]

        for i, h in enumerate(history_):
            role = "user" if i % 2 == 0 else "assistant"
            messages += [{"role": role, "content": h}]

        try:
            # 检查是否为 boyin 模型
            if self._is_boyin_model():
                # 使用新的send_request方法
                response_data = self.send_request(messages)
                if not response_data:
                    raise Exception("API call failed: No response data returned")
                return response_data.get("output", {}).get("text", "")
            else:
                # 原有的调用逻辑
                call_params = self.chat_kwargs.copy()
                response = self.client.chat.completions.create(
                    messages=messages,
                    **call_params
                )
                return _convert_openai_response_to_response(response).message.content
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            # 对于所有模型的错误处理都使用原来的方式
            import time
            time.sleep(1)  # Wait 1 second before retry
            if self._is_boyin_model():
                # 再次尝试使用新的send_request方法
                response_data = self.send_request(messages)
                if not response_data:
                    raise Exception("API call failed: No response data returned")
                return response_data.get("output", {}).get("text", "")
            else:
                # 原有调用的重试逻辑
                call_params = self.chat_kwargs.copy()
                response = self.client.chat.completions.create(
                    messages=messages,
                    **call_params
                )
                return _convert_openai_response_to_response(response).message.content

    def simple_chat_reasoning(self, query: str, history: Optional[List[str]] = None, sys_prompt: str = "", debug: bool = False) -> Any:
        """Simple interface for chat with history support."""
        if debug:
            # Save input arguments to cache
            cache_data = {
                'query': query,
                'history': history,
                'sys_prompt': sys_prompt
            }
            date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs(f'log/llamakit', exist_ok=True)
            with open(f'log/llamakit/cache-{date_time}.pkl', 'wb') as f:
                pickle.dump(cache_data, f)
                
        if "boyin" in self.model or "cong" in self.model:
            messages = []
        else:
            messages = [{"role": "system", "content": sys_prompt}]

        if history is None:
            history_ = []
        else:
            history_ = history.copy()
        history_ += [query]

        for i, h in enumerate(history_):
            role = "user" if i % 2 == 0 else "assistant"
            messages += [{"role": role, "content": h}]

        try:
            # 检查是否为 boyin 模型
            if self._is_boyin_model():
                # 使用新的send_request方法
                response_data = self.send_request(messages)
                if not response_data:
                    raise Exception("API call failed: No response data returned")
                return response_data.get("output", {}).get("text", "")
            else:
                # 原有的调用逻辑
                call_params = self.chat_kwargs.copy()
                call_params["stream"] = True
                if "qwen3" not in self.model:
                    call_params["temperature"] = 0.7

                # if "qwen3" in self.model:
                #     call_params["extra_body"] = {"enable_thinking": False}

                completion = self.client.chat.completions.create(
                    messages=messages,
                    **call_params
                )

                ans = ""
                enter_think = False
                leave_think = False
                for chunk in completion:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                            if not enter_think:
                                enter_think = True
                                ans += "<think>"
                            ans += delta.reasoning_content
                        elif delta.content:
                            if enter_think and not leave_think:
                                leave_think = True
                                ans += "</think>"
                            ans += delta.content

                return ans
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            # 对于所有模型的错误处理都使用原来的方式
            import time
            time.sleep(1)  # Wait 1 second before retry
            if self._is_boyin_model():
                # 再次尝试使用新的send_request方法
                response_data = self.send_request(messages)
                if not response_data:
                    raise Exception("API call failed: No response data returned")
                return response_data.get("output", {}).get("text", "")
            else:
                # 原有调用的重试逻辑
                call_params = self.chat_kwargs.copy()
                call_params["stream"] = True
                if "qwen3" not in self.model:
                    call_params["temperature"] = 0.7

                # if "qwen3" in self.model:
                #     call_params["extra_body"] = {"enable_thinking": False}

                completion = self.client.chat.completions.create(
                    messages=messages,
                    **call_params
                )

                ans = ""
                enter_think = False
                leave_think = False
                for chunk in completion:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                            if not enter_think:
                                enter_think = True
                                ans += "<think>"
                            ans += delta.reasoning_content
                        elif delta.content:
                            if enter_think and not leave_think:
                                leave_think = True
                                ans += "</think>"
                            ans += delta.content

                return ans


def test_all_models(api_key: str, prompt: str = "What is quantum computing?") -> None:
    """Test all available ECONML models with a simple prompt.
    
    Args:
        api_key: The ECONML API key
        prompt: The test prompt to use
    """
    print("\n=== Testing All ECONML Models ===\n")

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content=prompt)
    ]

    for model in EconmlLLM.AVAILABLE_MODELS:
        print(f"\nTesting model: {model}")
        print("-" * 50)

        try:
            llm = EconmlLLM(econml_api_key=api_key, model=model)
            response = llm.chat(messages)
            print(f"Response: {response.message.content[:200]}...")  # Print first 200 chars

            # Print token usage if available
            if "token_usage" in response.additional_kwargs:
                usage = response.additional_kwargs["token_usage"]
                print(f"\nToken usage: {usage}")

        except Exception as e:
            print(f"Error testing {model}: {str(e)}")

        print("-" * 50)


if __name__ == "__main__":
    # Boyin模型使用示例
    api_key = os.getenv("BOYIN_API_KEY")
    if not api_key:
        raise ValueError("BOYIN_API_KEY environment variable is not set. Please set it before running the test.")
    
    print("\n=== 测试 Boyin 模型 ===\n")
    
    try:
        # 初始化Boyin模型
        llm = EconmlLLM(model="claude-3-7-sonnet")
        
        # 简单问答示例（无历史记录）
        query ="""\n您是一位资深的理财内容评估专家，负责对两位理财师（理财师A和理财师B）的回答进行评分和比较。
        两位理财师均以\"蚂小财\"的身份回复用户问题，目标是通过轻松幽默、通俗易懂的方式为用户提供专业且个性化的理财建议。
        \n\n#输入\n##对话历史\n\n\n##用户问题：\n你好\n\n##理财师A回答：\n**🎉 Hi~ 我是你的支付宝理财小助手蚂小财！** 
        \n💰 理财科普 | 💡 产品答疑 | 🛡️ 保险攻略 \n👇 随时问我，比如： \n- 余额宝收益怎么算？ \n- 基金定投怎么玩？ \n- 养老金怎么规划？ \n\n表情包三连击： \n(✪ω✪) 专业不烧脑 \n(•̀ω•́)✧ 人狠话不多 \n(๑¯◡¯๑) 陪你慢慢变富~\n\n##理财师B回答：\n你好呀~有什么财富管理的小困惑需要我帮忙解答吗？💰✨ 无论是理财小技巧还是资产配置问题，我随时在线为你支招哦～ (๑•̀ㅂ•́)و✧\n\n#评分细则\n对比两个AI理财师对用户问题的回答，评估标准包括:\n\n            1. 结论实用性：建议是否具体可行，能直接指导用户行动\n\n            2. 论据丰富度：是否提供充分的数据、案例或理由支持建议\n\n            3. 表达易懂度：是否用通俗易懂的语言解释专业概念\n\n            4. 有据可依性：建议和分析是否基于可靠的数据和事实\n\n            5. 个性化定制：是否根据用户具体情况提供针对性建议\n\n            6. 情绪回应：是否恰当回应用户情绪需求，建立共情连接\n\n            优质回答应在以上各方面都表现良好，既要专业可靠，又要贴近用户。\n            \n\n#执行action：\n请严格遵循评分细则，综合比较理财师A和理财师B的回答质量。分析两位理财师在各个维度的表现，
        并给出详细的评分原因。最后需要选出整体表现更好的一位理财师。\n\n输出格式：\n<decision>\n评分原因：\n[详细分析两位理财师的回答优劣，包括在专业性、易懂性、个性化等方面的表现]\n\n总体更优回答：<better>理财师A/理财师B</better>\n</decision>"""
        # print(f"用户问题: {query}")
        
        response = llm.simple_chat(
            query=query,
            sys_prompt="你是一个乐于助人的AI助手，请用中文回答问题。"
        )
        
        print("\n回答:")
        print(response)
        
    except Exception as e:
        print(f"调用Boyin模型失败: {str(e)}")


