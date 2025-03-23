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
    """
    Assemble an AgentGroupChat by creating ChatCompletionAgents,
    attaching plugins loaded via the Calibrator_Component_Plugin,
    and dynamically adding functions from the provided function_list.
    
    Args:
        group_chat_info (dict): Information about the chat group.
        agent_list (list): Agents info with keys like agent_name, agent_instruction, service_id.
        plugin_list (list): Each dict should include plugin_class_name (e.g., "sample2_components.Scrum_Champ")
                            and plugin_name.
        function_list (list): Each dict should include function_class_name (e.g., "sample2_components.search")
                              and description.
        azure_openai_endpoint (str): Endpoint URL for Azure OpenAI.
        azure_openai_model (str): Deployment/model name for Azure OpenAI.
    
    Returns:
        AgentGroupChat: The assembled multi-agent chat object.
    """

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

    agents = []
    # Agent1
    agent1_kernel = Kernel()
    agent1_chat_completion_service = AzureChatCompletion(
        service_id=agent_list[0]["service_id"],
        deployment_name=azure_openai_model,
        ad_token_provider = token_provider,
        async_client=azure_oai_client,
    )
    agent1_kernel.add_service(agent1_chat_completion_service)
    settings = agent1_kernel.get_prompt_execution_settings_from_service_id(service_id=agent_list[0]["service_id"])
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    # Create a new chat completion agent with the Azure OpenAI client
    chat_completion_agent = ChatCompletionAgent(
        kernel=agent1_kernel,
        name= agent_list[0]["agent_name"],
        service_id = agent_list[0]["service_id"],
        description=agent_list[0]["agent_description"],
        instructions=agent_list[0]["agent_instruction"],
    )
    agents.append(chat_completion_agent)

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



    # --- Define termination strategy for the group chat ---
    # Define a termination keyword
    TERMINATION_KEYWORD = "TERMINATE"
    
    # Create a termination function that evaluates whether to end the conversation.
    termination_function = KernelFunctionFromPrompt(
        function_name="termination_function",
        prompt="""You are a termination evaluator. Analyze the chat history and if the last agent response indicates that the task is complete, output the keyword 'TERMINATE'. Otherwise, output nothing.
Chat History:
{{$history}}
"""
    )
    
    # Create a separate kernel for termination and add an Azure OpenAI service for it.
    termination_kernel = Kernel()
    termination_chat_completion_service = AzureChatCompletion(
        service_id="termination",
        deployment_name=azure_openai_model,
        ad_token_provider=token_provider,
        async_client=azure_oai_client,
    )

    termination_kernel.add_service(termination_chat_completion_service)
    
    # Set the termination strategy using the termination function and kernel.
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
    selection_function = KernelFunctionFromPrompt(
        function_name="selection_function",
        prompt="""You are the multi-agent coordinator. Your task is to select exactly one agent for the next turn, based solely on the conversation history. 
            Allowed agents are provided in the variable "agents". 
            Rules:
            Output only the name of the selected agent. Chat History: {{$history}}
            Chat History:
            {{$history}}
        """
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

