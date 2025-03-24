# Copyright (c) Microsoft. All rights reserved.

from typing import List
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import KernelFunctionSelectionStrategy
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import KernelFunctionTerminationStrategy
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
import asyncio
from typing import Annotated
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI
import time
from sk_calibrator_component_plugin import Calibrator_Component_Plugin
from sk_calibrator_component_function import Calibrator_Component_Function



def AssembleAgentGroupChat(group_chat_info, agent_list, plugin_list, function_list, agent_topology, azure_openai_endpoint: str, azure_openai_model: str) -> AgentGroupChat:
    """
    Assemble an AgentGroupChat by creating ChatCompletionAgents based on the agent_topology,
    attaching plugins loaded via the Calibrator_Component_Plugin, and dynamically adding functions
    from the topology-specified functions.

    Args:
        group_chat_info (dict): Information about the chat group.
        agent_list (list): List of agent configurations.
        plugin_list (list): List of plugin configurations.
        function_list (list): List of function configurations.
        agent_topology (dict): Describes which agents include which plugins and their functions.
        azure_openai_endpoint (str): Endpoint URL for Azure OpenAI.
        azure_openai_model (str): Deployment/model name for Azure OpenAI.

    Returns:
        AgentGroupChat: The assembled multi-agent chat object.
    """
    agent_group_chat = AgentGroupChat()

    # Create a new Azure OpenAI client
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    azure_oai_client = AsyncAzureOpenAI(
        api_version="2025-01-01-preview",
        azure_endpoint=azure_openai_endpoint,
        azure_ad_token_provider=token_provider
    )

    agents = []
    # Iterate over each agent defined in the topology
    for topology_agent in agent_topology.get("agents", []):
        agent_name = topology_agent.get("agent_name")
        # Lookup agent configuration from agent_list
        agent_config = next((a for a in agent_list if a.get("agent_name") == agent_name), None)
        if agent_config is None:
            continue  # Or raise an error if strict matching is required

        # Create a new kernel and chat completion service for the agent
        agent_kernel = Kernel()
        service_id = agent_config.get("service_id")
        chat_completion_service = AzureChatCompletion(
            service_id=service_id,
            deployment_name=azure_openai_model,
            ad_token_provider=token_provider,
            async_client=azure_oai_client,
        )
        agent_kernel.add_service(chat_completion_service)
        settings = agent_kernel.get_prompt_execution_settings_from_service_id(service_id)
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

        # Create the ChatCompletionAgent
        chat_agent = ChatCompletionAgent(
            kernel=agent_kernel,
            name=agent_config.get("agent_name"),
            service_id=service_id,
            description=agent_config.get("agent_description"),
            instructions=agent_config.get("agent_instruction"),
        )
        agents.append(chat_agent)
        agent_group_chat.add_agent(chat_agent)

        # Process plugins for the current agent per the topology
        for topology_plugin in topology_agent.get("plugins", []):
            plugin_name = topology_plugin.get("plugin_name")
            # Lookup plugin configuration from plugin_list
            plugin_config = next((p for p in plugin_list if p.get("plugin_name") == plugin_name), None)
            if plugin_config is None:
                continue

            plugin_class_name = plugin_config.get("plugin_class_name")
            plugin_component = Calibrator_Component_Plugin(plugin_class_name)
            plugin_instance = plugin_component.get_plugin()
            plugin_class = plugin_component.get_plugin_class()

            # Add functions to the plugin
            for func in topology_plugin.get("functions", []):
                function_class_name = func.get("function_class_name")
                # Lookup function description from function_list if provided
                func_config = next((f for f in function_list if f.get("function_class_name") == function_class_name), {})
                function_description = func_config.get("description", "")
                function_component = Calibrator_Component_Function(function_class_name, function_description)
                function_kernel = function_component.get_function()
                # Derive the simple function name from the full function class string.
                _, function_name = function_class_name.rsplit('.', 1)
                setattr(plugin_class, function_name, function_kernel)

            # Attach the plugin to the agent's kernel.
            agent_kernel.add_plugin(plugin_instance, plugin_name=plugin_name)

    # --- Define termination strategy for the group chat ---
    TERMINATION_KEYWORD = "TERMINATE"
    termination_function_prompt = group_chat_info["termination_function_prompt"]
    termination_function = KernelFunctionFromPrompt(
        function_name="termination_function",
        prompt=termination_function_prompt
    )

    termination_kernel = Kernel()
    termination_chat_completion_service = AzureChatCompletion(
        service_id="termination",
        deployment_name=azure_openai_model,
        ad_token_provider=token_provider,
        async_client=azure_oai_client,
    )
    termination_kernel.add_service(termination_chat_completion_service)

    agent_group_chat.termination_strategy = KernelFunctionTerminationStrategy(
        agents=agents,
        function=termination_function,
        kernel=termination_kernel,
        result_parser=lambda result: TERMINATION_KEYWORD in str(result.value[0]),
        history_variable_name="history",
        maximum_iterations=7,
    )
    # --- End termination strategy definition ---

    # --- Define selection strategy for the group chat ---
    selection_function_prompt = group_chat_info["selection_function_prompt"]
    selection_function = KernelFunctionFromPrompt(
        function_name="selection_function",
        prompt=selection_function_prompt
    )

    selection_kernel = Kernel()
    selection_chat_completion_service = AzureChatCompletion(
        service_id="selection",
        deployment_name=azure_openai_model,
        ad_token_provider=token_provider,
        async_client=azure_oai_client,
    )
    selection_kernel.add_service(selection_chat_completion_service)

    agent_group_chat.selection_strategy = KernelFunctionSelectionStrategy(
        function=selection_function,
        kernel=selection_kernel,
        result_parser=lambda result: str(result.value[0]) if result.value is not None else "",
        agent_variable_name="agents",
        history_variable_name="history",
    )
    # --- End selection strategy definition ---

    return agent_group_chat

