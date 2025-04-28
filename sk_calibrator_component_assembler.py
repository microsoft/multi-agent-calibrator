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
import json
from typing import Annotated
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI
import time
from sk_calibrator_component_plugin import Calibrator_Component_Plugin
from sk_calibrator_component_function import Calibrator_Component_Function

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

def add_mcp_agent(agents: List[ChatCompletionAgent], azure_openai_model: str, azure_oai_client: Annotated[AsyncAzureOpenAI, "Async Azure OpenAI Client"], token_provider: Annotated[DefaultAzureCredential, "Azure AD Token Provider"]):
    """Add an agent connected to a local MCP server, exposing its tools to the agent."""
    # Configure the MCP server connection (launch local server via stdio)
    server_params = StdioServerParameters(
        command="python", 
        args=["sample2_mcp_server.py"],  # adjust path or args as needed
        env=None
    )
    # Asynchronous routine to start MCP session and list available tools
    async def _connect_and_discover():
        # Launch the MCP server and open stdio streams
        read_stream, write_stream = await stdio_client(server_params).__aenter__()
        session = ClientSession(read_stream, write_stream)
        await session.initialize()                     # Establish MCP handshake
        tools_response = await session.list_tools()    # Discover tools exposed by server
        return session, tools_response.tools

    # Run the async connection routine and get the session and tools (one-time setup)
    session, tools = asyncio.get_event_loop().run_until_complete(_connect_and_discover())
    # (The session remains open for subsequent tool invocations)

    # Create a new Semantic Kernel for this agent and add an Azure OpenAI chat service
    agent_kernel = Kernel()
    mcp_service_id = "mcp_service"
    mcp_chat_service = AzureChatCompletion(
        service_id=mcp_service_id,
        deployment_name=azure_openai_model,
        ad_token_provider=token_provider,
        async_client=azure_oai_client,
    )
    agent_kernel.add_service(mcp_chat_service)
    # Enable automatic function selection (function calling) for this agent’s LLM
    settings = agent_kernel.get_prompt_execution_settings_from_service_id(mcp_service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Dynamically create a plugin to hold MCP tool functions
    MCPPlugin = type("MCPPlugin", (), {})
    # Define initializer to store the MCP session
    def _init(self, session):
        self._session = session
    MCPPlugin.__init__ = _init

    # Convert each MCP tool into a Kernel Function and attach to the plugin
    for tool in tools:
        tool_name = tool.name
        tool_desc = tool.description or ""
        # Define an async function that calls the MCP tool via the session
        async def _tool_func(self, **kwargs):
            # Invoke the tool on the MCP session with provided arguments
            result = await self._session.call_tool(tool_name, kwargs)
            # Extract text content from the result (assuming text output)
            if result and result.content:
                # Return the text result to the agent
                return result.content[0].text
            return ""  # default empty string if no content returned

        # Mark this function as a kernel function with appropriate metadata
        decorated_func = kernel_function(name=tool_name, description=tool_desc)(_tool_func)
        # Attach the function to the MCPPlugin class
        setattr(MCPPlugin, tool_name, decorated_func)

    # Instantiate the plugin and register it with the agent’s kernel&#8203;:contentReference[oaicite:1]{index=1}
    mcp_plugin_instance = MCPPlugin(session)
    agent_kernel.add_plugin(mcp_plugin_instance, plugin_name="MCP")

    # Create the ChatCompletionAgent wired to use the MCP server’s tools
    mcp_agent = ChatCompletionAgent(
        kernel=agent_kernel,
        name="MCPAgent",
        service_id=mcp_service_id,
        description="Agent connected to local MCP server",
        instructions="You are an agent with access to various tools via MCP."
    )
    # Append the new agent to the agents list
    agents.append(mcp_agent)


#def AssembleAgentGroupChat(group_chat_info, agent_list, plugin_list, function_list, agent_topology, azure_openai_endpoint: str, azure_openai_model: str) -> AgentGroupChat:
def AssembleAgentGroupChat(sk_components) -> AgentGroupChat:
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

    #print(json.dumps(sk_components.to_dict(), indent=4))

    agent_group_chat = AgentGroupChat()

    # Create a new Azure OpenAI client
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    azure_oai_client = AsyncAzureOpenAI(
        api_version="2025-01-01-preview",
        azure_endpoint=sk_components.azure_openai_endpoint,
        azure_ad_token_provider=token_provider
    )

    agents = []
    # Iterate over each agent defined in the topology
    for topology_agent in sk_components.agent_topology.get("agents", []):
        agent_name = topology_agent.get("agent_name")
        # Lookup agent configuration from agent_list
        agent_config = next((a for a in sk_components.agent_list if a.get("agent_name") == agent_name), None)
        if agent_config is None:
            continue  # Or raise an error if strict matching is required

        # Create a new kernel and chat completion service for the agent
        agent_kernel = Kernel()
        service_id = agent_config.get("service_id")
        chat_completion_service = AzureChatCompletion(
            service_id=service_id,
            deployment_name=sk_components.azure_openai_model,
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
            plugin_config = next((p for p in sk_components.plugin_list if p.get("plugin_name") == plugin_name), None)
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
                func_config = next((f for f in sk_components.function_list if f.get("function_class_name") == function_class_name), {})
                function_description = func_config.get("description", "")
                function_component = Calibrator_Component_Function(function_class_name, function_description)
                function_kernel = function_component.get_function()
                # Derive the simple function name from the full function class string.
                _, function_name = function_class_name.rsplit('.', 1)
                setattr(plugin_class, function_name, function_kernel)

            # Attach the plugin to the agent's kernel.
            agent_kernel.add_plugin(plugin_instance, plugin_name=plugin_name)

    # Add a new weather agent to the group chat
    add_mcp_agent(agents, sk_components.azure_openai_model, azure_oai_client, token_provider)

    # --- Define termination strategy for the group chat ---
    TERMINATION_KEYWORD = "TERMINATE"
    termination_function_prompt = sk_components.group_chat_info["termination_function_prompt"]
    termination_function = KernelFunctionFromPrompt(
        function_name="termination_function",
        prompt=termination_function_prompt
    )

    termination_kernel = Kernel()
    termination_chat_completion_service = AzureChatCompletion(
        service_id="termination",
        deployment_name=sk_components.azure_openai_model,
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
    selection_function_prompt = sk_components.group_chat_info["selection_function_prompt"]
    selection_function = KernelFunctionFromPrompt(
        function_name="selection_function",
        prompt=selection_function_prompt
    )

    selection_kernel = Kernel()
    selection_chat_completion_service = AzureChatCompletion(
        service_id="selection",
        deployment_name=sk_components.azure_openai_model,
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

# Test
from sample2_component_list import sk_component_abstraction, group_chat_info, agent_list, plugin_list, function_list, agent_topology, azure_openai_endpoint, azure_openai_model
from sk_calibrator_component_assembler import AssembleAgentGroupChat
sk_components = sk_component_abstraction(group_chat_info, agent_list, plugin_list, function_list, agent_topology, azure_openai_endpoint, azure_openai_model)
multi_chat = AssembleAgentGroupChat(
    sk_components
)
print(multi_chat)