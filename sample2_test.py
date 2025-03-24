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

from sk_calibrator_component_assembler import AssembleAgentGroupChat

from sample2_component_list import agent_list, plugin_list, function_list, group_chat_info, azure_openai_endpoint,azure_openai_model

# Import the configuration loader
from sk_calibrator_config import load_config
# Load endpoints from YAML configuration
config = load_config("sample_sk_orchestrator_config.yaml")

AssembleAgentGroupChat(group_chat_info, agent_list,plugin_list, function_list, azure_openai_endpoint = azure_openai_endpoint, azure_openai_model = azure_openai_model)

async def async_output(user_input: str, chat_history_input: ChatHistory, azureopenai_endpoint: str) -> str:

    multi_chat = AssembleAgentGroupChat(group_chat_info, agent_list,plugin_list, function_list, azure_openai_endpoint = azureopenai_endpoint, azure_openai_model = "gpt-4o-mini-deploy")

    delta = ["agent1"]

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