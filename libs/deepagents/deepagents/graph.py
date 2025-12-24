"""Deepagents come with planning, filesystem, and subagents."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
# ---------------------------- 库内封装的模块 ----------------------------
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware


# ---------------------------- 基础提示词 用于控制文件管理todolist等模块 ----------------------------
BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


def get_default_model() -> ChatAnthropic:
    """Get the default model for deep agents.

    Returns:
        ChatAnthropic instance configured with Claude Sonnet 4.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,
    )


def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
    exclude_tools: list[str] | None = None,
    subagent_exclude_tools: list[str] | None = None,
    summarization_config: dict[str, Any] | None = None,
) -> CompiledStateGraph:
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    seven file and execution tools: ls, read_file, write_file, edit_file, glob, grep, execute,
    and a tool to call subagents.

    The execute tool allows running shell commands if the backend implements SandboxBackendProtocol.
    For non-sandbox backends, the execute tool will return an error message.

    Args:
        model: The model to use. Defaults to Claude Sonnet 4.
        tools: The tools the agent should have access to.
        system_prompt: The additional instructions the agent should have. Will go in
            the system prompt.
        middleware: Additional middleware to apply after standard middleware.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the
                  sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (either a LanguageModelLike instance or dict
                  settings)
                - (optional) `middleware` (list of AgentMiddleware)
        response_format: A structured output response format to use for the agent.
        context_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        store: Optional store for persistent storage (required if backend uses StoreBackend).
        backend: Optional backend for file storage and execution. Pass either a Backend instance
            or a callable factory like `lambda rt: StateBackend(rt)`. For execution support,
            use a backend that implements SandboxBackendProtocol.
        interrupt_on: Optional Dict[str, bool | InterruptOnConfig] mapping tool names to
            interrupt configs.
        debug: Whether to enable debug mode. Passed through to create_agent.
        name: The name of the agent. Passed through to create_agent.
        cache: The cache to use for the agent. Passed through to create_agent.
        exclude_tools: List of default system tool names to exclude from the main agent.
            Available tools: `write_todos`, `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`, `task`.
        subagent_exclude_tools: List of default system tool names to exclude from all 
            subagents. Available tools: `write_todos`, `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`, `task`.
        summarization_config: Configuration for the context summarization middleware.
            - `max_tokens`: (int) The token limit before summarization is triggered (e.g., 100000).
            - `keep_messages`: (int) How many recent messages to keep as-is (e.g., 5).
            - (Advanced) `trigger`: Alternative tuple format, e.g. `("fraction", 0.8)`.
            - (Advanced) `keep`: Alternative tuple format, e.g. `("fraction", 0.1)`.
            If not provided, defaults are intelligently calculated based on model profile.

    Returns:
        A configured deep agent.
    """
    if model is None:
        model = get_default_model()

    # Support both intuitive and advanced keys
    trigger_override = None
    keep_override = None
    if summarization_config:
        # Priority 1: Direct token/message count
        if "max_tokens" in summarization_config:
            trigger_override = ("tokens", summarization_config["max_tokens"])
        elif "trigger" in summarization_config:
            trigger_override = summarization_config["trigger"]
            
        if "keep_messages" in summarization_config:
            keep_override = ("messages", summarization_config["keep_messages"])
        elif "keep" in summarization_config:
            keep_override = summarization_config["keep"]

    if (
        model.profile is not None
        and isinstance(model.profile, dict)
        and "max_input_tokens" in model.profile
        and isinstance(model.profile["max_input_tokens"], int)
    ):
        trigger = trigger_override or ("fraction", 0.85)
        keep = keep_override or ("fraction", 0.10)
    else:
        trigger = trigger_override or ("tokens", 150000)
        keep = keep_override or ("messages", 6)




    main_exclude = exclude_tools or []
    sub_exclude = subagent_exclude_tools or []

    def filter_tools(mw: AgentMiddleware, exclude_list: list[str]) -> AgentMiddleware:
        if hasattr(mw, "tools"):
            mw.tools = [
                t for t in mw.tools 
                if (t.name if hasattr(t, "name") else t.get("name")) not in exclude_list
            ]
        return mw

    deepagent_middleware = [
        filter_tools(TodoListMiddleware(), main_exclude),
        filter_tools(FilesystemMiddleware(backend=backend), main_exclude),
        filter_tools(
            SubAgentMiddleware(
                default_model=model,
                default_tools=tools,
                subagents=subagents if subagents is not None else [],
                default_middleware=[
                    filter_tools(TodoListMiddleware(), sub_exclude),
                    filter_tools(FilesystemMiddleware(backend=backend), sub_exclude),
                    SummarizationMiddleware(
                        model=model,
                        trigger=trigger,
                        keep=keep,
                        trim_tokens_to_summarize=None,
                    ),
                    AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                    PatchToolCallsMiddleware(),
                ],
                default_interrupt_on=interrupt_on,
                general_purpose_agent=True,
            ),
            main_exclude
        ),
        SummarizationMiddleware(
            model=model,
            trigger=trigger,
            keep=keep,
            trim_tokens_to_summarize=None,
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]

    print(f"DEBUG: Loading {len(deepagent_middleware)} middlewares for agent: {name}")
    if middleware:
        deepagent_middleware.extend(middleware)
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return create_agent(
        model,
        system_prompt=system_prompt + "\n\n" + BASE_AGENT_PROMPT if system_prompt else BASE_AGENT_PROMPT,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
