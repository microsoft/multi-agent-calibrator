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



def AssembleAgentGroupChat(group_chat_info, agent_list, plugin_list, function_list, azure_openai_endpoint: str, azure_openai_model: str) -> AgentGroupChat:


    # Create a new agent group chat
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

    # Agent1
    agent1_kernel = Kernel()
    # Create a new chat completion agent with the Azure OpenAI client
    chat_completion_agent = ChatCompletionAgent(
        kernel=agent1_kernel,
        name= agent_list[0]["agent_name"],
        service_id = agent_list[0]["service_id"],
        description=agent_list[0]["agent_description"],
        instructions=agent_list[0]["agent_instruction"],
    )

    # Add the chat completion agent to the agent group chat
    agent_group_chat.add_agent(chat_completion_agent)

    plugin_class_name_1 = plugin_list[0]["plugin_class_name"]
    plugin_name_1 = plugin_list[0]["plugin_name"]
    plugin_component1 = Calibrator_Component_Plugin(plugin_class_name_1)  # Create an instance of the plugin class
    plugin1 = plugin_component1.get_plugin()
    plugin_class1 = plugin_component1.get_plugin_class()

    function_class_name1= function_list[0]["function_class_name"]
    function_class_description1 = function_list[0]["description"]
    function_module_name1, function_name1 = function_class_name1.rsplit('.', 1)
    function_component1 = Calibrator_Component_Function(function_class_name1, function_class_description1)
    function1_kernel = function_component1.get_function()
    setattr(plugin_class1, function_name1, function1_kernel)

    agent1_kernel.add_plugin(plugin1, plugin_name=plugin_name_1)


    return agent_group_chat

