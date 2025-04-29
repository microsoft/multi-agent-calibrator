# Copyright (c) Microsoft. All rights reserved.
"""
Refactored to integrate the MCP server **exactly the same way** as the working
sample provided by the user, i.e. via **`MCPStdioPlugin`** instead of the manual
`ClientSession` wrapper.  Only the MCP‑related pieces changed; the rest of the
structure (plugins, selection/termination strategies, etc.) stays the same.

Key changes
-----------
1. **`add_mcp_agent`** now:
   • Uses `MCPStdioPlugin` to launch the local server (`sample2_mcp_server.py`).
   • Returns **both** the new `ChatCompletionAgent` _and_ the plugin instance so
     callers can shut the plugin down with `await plugin.__aexit__(...)`.
2. **`AssembleAgentGroupChat`** collects the plugin and returns it alongside the
   `AgentGroupChat` so the caller can manage its lifetime (mirrors the hint’s
   pattern).
3. The file’s test runner shows proper cleanup of the MCP plugin.

NOTE: If you already imported this module elsewhere, update the call site to
unpack the returned `(multi_chat, mcp_plugin)` tuple.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack
from typing import List, Tuple

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI

from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import (
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.prompt_execution_settings import (
    PromptExecutionSettings,
)
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.functions.kernel_function_from_prompt import (
    KernelFunctionFromPrompt,
)
from semantic_kernel.kernel import Kernel

from sk_calibrator_component_plugin import Calibrator_Component_Plugin
from sk_calibrator_component_function import Calibrator_Component_Function

# ---------------------------------------------------------------------------
# MCP agent helper -----------------------------------------------------------
# ---------------------------------------------------------------------------


async def add_mcp_agent(
    token_provider,
    azure_oai_client,
    azure_openai_model: str,
    *,
    server_script: str = "sample2_mcp/sample2_mcp_server.py",
    agent_name: str = "weather_agent",
) -> Tuple[ChatCompletionAgent, MCPStdioPlugin]:
    """Spin up the local MCP server via *MCPStdioPlugin* and wrap it in an agent."""

    # Launch server and obtain plugin (same pattern as the hint)
    weather_plugin = await MCPStdioPlugin(
        name="weather",
        description="Fetch weather alerts & forecast",
        command="python",
        args=[server_script],
    ).__aenter__()

    # Build kernel + chat service for this agent
    agent_kernel = Kernel()
    service_id = "mcp_service"
    chat_service = AzureChatCompletion(
        service_id=service_id,
        deployment_name=azure_openai_model,
        ad_token_provider=token_provider,
        async_client=azure_oai_client,
    )
    agent_kernel.add_service(chat_service)
    settings: PromptExecutionSettings = agent_kernel.get_prompt_execution_settings_from_service_id(
        service_id
    )
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Attach the MCP plugin to the kernel
    agent_kernel.add_plugin(
        plugin=weather_plugin,
        plugin_name="Weather_Plugin",
        description="Local MCP weather tool",
    )

    weather_agent = ChatCompletionAgent(
        kernel=agent_kernel,
        name=agent_name,
        arguments=KernelArguments(settings=settings),
        description="Use Weather_Plugin.get_alerts or get_forecast",
        instructions="When asked about weather, call the plugin’s tools.",
    )

    return weather_agent, weather_plugin


# ---------------------------------------------------------------------------
# Main assembly function -----------------------------------------------------
# ---------------------------------------------------------------------------


async def AssembleAgentGroupChat(sk_components) -> Tuple[AgentGroupChat, MCPStdioPlugin]:
    """Assemble the `AgentGroupChat` and return it **plus** the MCP plugin."""

    agent_group_chat = AgentGroupChat()

    # Shared Azure OpenAI client & token provider
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    azure_oai_client = AsyncAzureOpenAI(
        api_version="2025-01-01-preview",
        azure_endpoint=sk_components.azure_openai_endpoint,
        azure_ad_token_provider=token_provider,
    )

    agents: List[ChatCompletionAgent] = []

    # ---------------------------------------------------------------------
    # Build normal (non‑MCP) agents from topology -------------------------
    # ---------------------------------------------------------------------
    for topology_agent in sk_components.agent_topology.get("agents", []):
        agent_name = topology_agent.get("agent_name")
        agent_config = next(
            (a for a in sk_components.agent_list if a["agent_name"] == agent_name),
            None,
        )
        if agent_config is None:
            continue

        agent_kernel = Kernel()
        service_id = agent_config["service_id"]
        chat_service = AzureChatCompletion(
            service_id=service_id,
            deployment_name=sk_components.azure_openai_model,
            ad_token_provider=token_provider,
            async_client=azure_oai_client,
        )
        agent_kernel.add_service(chat_service)
        settings: PromptExecutionSettings = agent_kernel.get_prompt_execution_settings_from_service_id(
            service_id
        )
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

        chat_agent = ChatCompletionAgent(
            kernel=agent_kernel,
            name=agent_name,
            description=agent_config["agent_description"],
            instructions=agent_config["agent_instruction"],
            arguments=KernelArguments(settings=settings),
        )

        # Attach plugins & functions listed in topology
        for topology_plugin in topology_agent.get("plugins", []):
            plugin_name = topology_plugin["plugin_name"]
            plugin_config = next(
                (p for p in sk_components.plugin_list if p["plugin_name"] == plugin_name),
                None,
            )
            if plugin_config is None:
                continue

            plugin_class_name = plugin_config["plugin_class_name"]
            plugin_component = Calibrator_Component_Plugin(plugin_class_name)
            plugin_instance = plugin_component.get_plugin()
            plugin_class = plugin_component.get_plugin_class()

            for func in topology_plugin.get("functions", []):
                function_class_name = func["function_class_name"]
                func_config = next(
                    (
                        f
                        for f in sk_components.function_list
                        if f["function_class_name"] == function_class_name
                    ),
                    {},
                )
                function_description = func_config.get("description", "")
                function_component = Calibrator_Component_Function(
                    function_class_name, function_description
                )
                function_kernel = function_component.get_function()
                _, function_name = function_class_name.rsplit(".", 1)
                setattr(plugin_class, function_name, function_kernel)

            agent_kernel.add_plugin(plugin_instance, plugin_name=plugin_name)

        agents.append(chat_agent)
        #agent_group_chat.add_agent(chat_agent)

    # ---------------------------------------------------------------------
    # Add MCP agent (weather) ---------------------------------------------
    # ---------------------------------------------------------------------
    mcp_agent, mcp_plugin = await add_mcp_agent(
        token_provider,
        azure_oai_client,
        sk_components.azure_openai_model,
    )
    #agents.append(mcp_agent)
    agents = [mcp_agent]  # replace all agents with MCP agent
    agent_group_chat.add_agent(mcp_agent)

    # ---------------------------------------------------------------------
    # Selection & termination strategies ----------------------------------
    # ---------------------------------------------------------------------
    TERMINATION_KEYWORD = "TERMINATE"

    termination_function = KernelFunctionFromPrompt(
        function_name="termination_function",
        prompt=sk_components.group_chat_info["termination_function_prompt"],
    )
    termination_kernel = Kernel()
    termination_kernel.add_service(
        AzureChatCompletion(
            service_id="termination",
            deployment_name=sk_components.azure_openai_model,
            ad_token_provider=token_provider,
            async_client=azure_oai_client,
        )
    )

    selection_function = KernelFunctionFromPrompt(
        function_name="selection_function",
        prompt=sk_components.group_chat_info["selection_function_prompt"],
    )
    selection_kernel = Kernel()
    selection_kernel.add_service(
        AzureChatCompletion(
            service_id="selection",
            deployment_name=sk_components.azure_openai_model,
            ad_token_provider=token_provider,
            async_client=azure_oai_client,
        )
    )

    agent_group_chat.termination_strategy = KernelFunctionTerminationStrategy(
        agents=agents,
        function=termination_function,
        kernel=termination_kernel,
        result_parser=lambda result: TERMINATION_KEYWORD in str(result.value[0]),
        history_variable_name="history",
        maximum_iterations=7,
    )

    agent_group_chat.selection_strategy = KernelFunctionSelectionStrategy(
        function=selection_function,
        kernel=selection_kernel,
        result_parser=lambda result: str(result.value[0]) if result.value else "",
        agent_variable_name="agents",
        history_variable_name="history",
    )

    return agent_group_chat, mcp_plugin


# ---------------------------------------------------------------------------
# Local test/debug -----------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    async def _main():
        from sample2_component_list import (
            sk_component_abstraction,
            group_chat_info,
            agent_list,
            plugin_list,
            function_list,
            agent_topology,
            azure_openai_endpoint,
            azure_openai_model,
        )

        sk_components = sk_component_abstraction(
            group_chat_info,
            agent_list,
            plugin_list,
            function_list,
            agent_topology,
            azure_openai_endpoint,
            azure_openai_model,
        )

        multi_chat, mcp_plugin = await AssembleAgentGroupChat(sk_components)

        # simple interaction
        await multi_chat.add_chat_message(
            ChatMessageContent(role=AuthorRole.USER, content="Weather in Seattle?")
        )
        async for msg in multi_chat.invoke():
            print("\n", msg.name, "->", msg.content)
            if msg.name == "weather_agent":
                break

        # graceful shutdown of MCP plugin
        try:
            await mcp_plugin.__aexit__(None, None, None)
        except RuntimeError:
            pass

    asyncio.run(_main())
