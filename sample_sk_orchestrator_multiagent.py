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

MODEL_GPT_4O_MINI = "gpt-4o-mini-deploy"
MODEL_O1_MINI = "o1-mini-deploy"
azure_ai_model = MODEL_GPT_4O_MINI

MODEL_VERSION_GPT_4O = "2024-08-01-preview"
MODEL_VERSION_GPT_4O_MINI = "2025-01-01-preview"
azure_ai_api_version = MODEL_VERSION_GPT_4O

# Import the configuration loader
from sk_calibrator_config import load_config
# Load endpoints from YAML configuration
config = load_config("sample_sk_orchestrator_config.yaml")


class Scrum_Champ:
    def __init__(self):
        pass

    @kernel_function(description="Check the capacity available for the whole team")
    def search(self, user_question_or_keywords: str) -> List[str]:
        knowledge_str = "The team has 3 developers, who works 40 hours a week."
        return (knowledge_str)

def define_multi_agent(user_input: str, chat_history_input: ChatHistory, azureopenai_endpoint: str) -> AgentGroupChat:

    # Scrum Champ Agent
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    azure_oai_client = AsyncAzureOpenAI(
        api_version=azure_ai_api_version, 
        azure_endpoint=azureopenai_endpoint,
        azure_ad_token_provider=token_provider
    )

    scrum_champ_kernel = Kernel()
    service_id = "scrum_champ_agent_service"

    scrum_champ_chat_completion_service = AzureChatCompletion(
        service_id=service_id,
        ad_token_provider = token_provider,
        deployment_name=azure_ai_model,
        async_client=azure_oai_client,
    )
    scrum_champ_kernel.add_service(scrum_champ_chat_completion_service)
    settings = scrum_champ_kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    scrum_champ_agent = ChatCompletionAgent(
        service_id=service_id,
        kernel=scrum_champ_kernel,
        name="scrum_champ_agent",
        execution_settings=settings,
        description="TODO",
        instructions="""
            TODO:

            User Input:
            {user_input}

            Chat History:
            {chat_history_input}

            {{{{$history}}}}

            """,
    )
    scrum_champ_kernel.add_plugin(plugin=Scrum_Champ(), plugin_name="Scrum_Champ_Plugin", description="TODO")
    
    # UX Agent
    ux_kernel = Kernel()
    service_id = "ux_service"
    ux_chat_completion_service = AzureChatCompletion(
        service_id=service_id,
        deployment_name=azure_ai_model,
        ad_token_provider = token_provider,
        async_client=azure_oai_client,
    )
    ux_kernel.add_service(ux_chat_completion_service)
    settings = ux_kernel.get_prompt_execution_settings_from_service_id(service_id="ux_service")
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    ux_agent = ChatCompletionAgent(
        service_id=service_id,
        kernel=ux_kernel,
        name="ux_agent",
        execution_settings=settings,
        description="You are an User Experience expert. You help polish the output from previous Agent to better readable result for Copilot chat experience.",
        instructions="""
            Your role is to refine responses to ensure they are clear, concise, and user-friendly. 

            """,
    )

    # Multi-Agent Chat
    # Scrum champ agent help generate a sprint plan
    AGENT_SCRUM_CHAMP = "scrum_champ_agent" 
    # UX Agent help to format agent UX experience
    AGENT_UX = "ux_agent" # UX

    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
    You are the multi-agent coordinator. Your task is to select exactly one agent from the list below for the next turn, based solely on the conversation history.

    Allowed agents (in order of execution):
    - {AGENT_SCRUM_CHAMP}
    - {AGENT_UX}

    Rules:
    1. Call each agent exactly once.    
    2. Output only the name of the selected agent, nothing else.

    User Input:
    {user_input}

    Chat History:
    {chat_history_input}

    {{{{$history}}}}
    """
    )

    TERMINATION_KEYWORD = "TERMINATE"

    termination_function = KernelFunctionFromPrompt(
        function_name="termination_function",
        prompt=f"""
    If last reply is by {AGENT_UX}. End the loop.

    RESPONSE:
    {{{{$history}}}}
    """
    )

    termination_kernel = Kernel()

    termination_chat_completion_service = AzureChatCompletion(
        service_id="termination",
        deployment_name=azure_ai_model,
        ad_token_provider = token_provider,
        async_client=azure_oai_client,
    )
    termination_kernel.add_service(termination_chat_completion_service)

    selection_kernel = Kernel()
    selection_chat_completion_service = AzureChatCompletion(
        service_id="selection",
        deployment_name=azure_ai_model,
        ad_token_provider = token_provider,
        async_client=azure_oai_client,
        )
    selection_kernel.add_service(selection_chat_completion_service)

    multi_chat = AgentGroupChat(
        agents=[scrum_champ_agent, ux_agent], 
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[scrum_champ_agent, ux_agent], 
            function=termination_function,
            kernel=termination_kernel,
            result_parser=lambda result: TERMINATION_KEYWORD in str(result.value[0]),

            history_variable_name="history",
            maximum_iterations=7,
        ),
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection_function,
            kernel=selection_kernel,
            result_parser=lambda result: str(result.value[0]) if result.value is not None else print('TERMINATE'),
            agent_variable_name="agents",
            history_variable_name="history",
        ),
    )

    return multi_chat

async def async_output(user_input: str, chat_history_input: ChatHistory, azureopenai_endpoint: str) -> str:

    multi_chat = define_multi_agent(user_input, chat_history_input, azureopenai_endpoint)

    delta = ["ux_agent"]

    if True:
        question_2 = user_input #messages[0]
        responses = []
        await multi_chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=question_2))

        TERMINATION_KEYWORD = "TERMINATE"  # Define your termination keyword

        last_response = ""

        # Variable to accumulate all agent outputs.
        aggregated_outputs = ""

        try:

            async for response in multi_chat.invoke():
                print("\n-------------------------------------\n", response.name, "\n",  response.content, "\n\n\n")
                last_response = response.content

                # Append the response to the aggregated outputs.
                aggregated_outputs += f"{response.name}: {response.content}\n"

                if response.name in delta:
                    responses.append(response.content)
                    responses.append("*" * 50)

                # Once the conversation ends, stop the loop
                print("Is the conversation complete: ", multi_chat.is_complete)
                if multi_chat.is_complete :
                    break

                # Check for termination keyword
                if TERMINATION_KEYWORD in response.content:
                    print("\n--- Conversation Ended by Agent ---")
                    break  # Exit the loop when the termination condition is met

        except Exception as e:
            print(f"Error: {e}")

        if len(responses) == 0:
            responses.append(last_response)
            responses.append("*" * 50)

        return "\n".join(responses)

async def main(user_input: str, chat_history_input: ChatHistory, azureopenai_endpoint: str):

    # Measure the calculation time of the async_output function
    start_time = time.time()
    response = await async_output(user_input,chat_history_input, azureopenai_endpoint)
    end_time = time.time()

    # Print the result
    print("\n\n\nUser Question: ", user_input)
    print("----- Multi-Agent Response Start -----")
    print(response)
    print("----- Multi-Agent Response End -----")

    # Print the time taken
    print(f"Time taken to get async_output: {end_time - start_time:.4f} seconds")

    return response


# Debug only
if __name__ == "__main__":
    user_input = f"Help me to make a sprint plan to create a library management system"
    chat_history_input = ChatHistory()
    azureopenai_endpoint = config.get("azureopenai_endpoint")
    asyncio.run(main(user_input, chat_history_input, azureopenai_endpoint))