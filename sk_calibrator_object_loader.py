# Copyright (c) Microsoft. All rights reserved.

from semantic_kernel.agents import AgentGroupChat
from sample_sk_orchestrator_multiagent import define_multi_agent

import asyncio
from pydantic import BaseModel
import json
import os
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.contents.chat_history import ChatHistory

from sk_calibrator_testcase import read_testcases_from_jsonl, convert_chat_history_to_sk_chat_history
from tool_aoai import call_aoai_withoutowndata_o1

# Import the configuration loader
from sk_calibrator_config import load_config
# Load endpoints from YAML configuration
config = load_config()

def load_group_chat():
    # TODO: Please implement your own based on your project

    azureopenai_endpoint = config.get("azureopenai_endpoint")

    test_questions = read_testcases_from_jsonl('./sample_sk_orchestrator_testcases.jsonl')

    # Test case 
    for question_01 in test_questions:
        user_input = question_01.question
        chat_history_input = convert_chat_history_to_sk_chat_history(question_01.chat_history)

        multi_agent = define_multi_agent(user_input, chat_history_input, azureopenai_endpoint = azureopenai_endpoint)
        return multi_agent
    

def decode_multi_agent(multi_chat: AgentGroupChat):
    
    """
    Inspect and print out the internal structure of the AgentGroupChat instance.
    This includes:
      - Selection Function (name, prompt, etc.)
      - Termination Function (name, prompt, etc.)
      - Each Agent's details (name, description, service_id, instructions)
      - Each Agent's Plugins (name, description, available functions, etc.)
    """

    # --- 1. Selection Strategy ---
    if multi_chat.selection_strategy and multi_chat.selection_strategy.function:
        selection_fn = multi_chat.selection_strategy.function
        print("=== Selection Function Details ===")
        print("Function Name: ", selection_fn.name)
        print("Function Prompt: ")
        # Safely check if the function has a 'prompt' or 'prompt_template' attribute
        if hasattr(selection_fn, "prompt"):
            print("Function Prompt:")
            print(selection_fn.prompt)
        elif hasattr(selection_fn, "prompt_template"):
            pt = selection_fn.prompt_template
            # Check if the object has 'prompt_template_config'
            if hasattr(pt, "prompt_template_config"):
                print("== Template Content ==")
                # Access the .template field
                print(pt.prompt_template_config.template)
            else:
                print("No prompt_template_config found.")
        else:
            print("No prompt_template found.")
        print("==================================\n")
    else:
        print("No Selection Strategy or Function found.\n")

    # --- 2. Termination Strategy ---
    if multi_chat.termination_strategy and multi_chat.termination_strategy.function:
        termination_fn = multi_chat.termination_strategy.function
        print("=== Termination Function Details ===")
        print("Function Name:", termination_fn.name)

        # Check if 'termination_fn' has a direct 'prompt' attribute
        if hasattr(termination_fn, "prompt"):
            print("Function Prompt:")
            print(termination_fn.prompt)
        # Otherwise, see if it has 'prompt_template' with a 'prompt_template_config.template'
        elif hasattr(termination_fn, "prompt_template"):
            pt = termination_fn.prompt_template
            if hasattr(pt, "prompt_template_config") and hasattr(pt.prompt_template_config, "template"):
                print("Function Prompt Template:")
                print(pt.prompt_template_config.template)
            else:
                print("No prompt_template_config or template found in termination function.")
        else:
            print("No prompt or prompt_template property found for termination function.")

        print("====================================\n")
    else:
        print("No Termination Strategy or Function found.\n")

    # --- 3. Agents ---
    print("=== Agents Details ===")
    for idx, agent in enumerate(multi_chat.agents, start=1):
        print(f"--- Agent #{idx} ---")
        print("Name:        ", agent.name)
        print("Description: ", agent.description)
        print("Service ID:  ", agent.service_id)
        print("Instructions:")
        print(agent.instructions)
        print()

        # --- 4. Agent's Plugins ---
        if agent.kernel.plugins:
            print("  -> Plugins for agent:", agent.name)
            for plugin_name, plugin in agent.kernel.plugins.items():
                print(f"     Plugin Name:        {plugin.name}")
                print(f"     Plugin Description: {plugin.description}")

                print(plugin.functions)


                if plugin.functions:
                    print("     Functions:")
                    for fn_name, fn_obj in plugin.functions.items():
                        print(f"        Function Name: {fn_obj.name}")
                        print(f"        Fully Qualified Name: {fn_obj.fully_qualified_name}")
                        print(f"        Description: {fn_obj.description}")
                    print()

            

        else:
            print("  -> No plugins found for this agent.")
        print("----------------------------------------\n")

    print("=== End of Multi-Agent Structure ===")


import json
from semantic_kernel.agents import AgentGroupChat

import json
import os
from semantic_kernel.agents import AgentGroupChat


def convert_multi_agent_to_json(multi_chat: AgentGroupChat) -> str:
    """
    Convert the AgentGroupChat structure into a JSON string following
    the same nested hierarchy that modify_multi_agent uses to navigate
    and modify elements.

    This includes:
      - selection_strategy.function:  (name, prompt, prompt_template, etc.)
      - termination_strategy.function: (name, prompt, prompt_template, etc.)
      - agents (list of agents)
         - name, description, service_id, instructions
         - plugins (list)
             - plugin name, description
             - functions (list)
                 - name, fully_qualified_name, description
    """

    def function_to_dict(fn):
        """
        Helper to convert a KernelFunctionFromPrompt or similar object
        into a nested dictionary structure that can reflect:
          function.name
          function.prompt
          function.prompt_template.prompt_template_config.template
        """
        fn_dict = {}
        if not fn:
            return fn_dict

        # Store function name
        if hasattr(fn, "name"):
            fn_dict["name"] = fn.name

        # Store prompt if it exists
        if hasattr(fn, "prompt"):
            fn_dict["prompt"] = fn.prompt

        # If there's a prompt_template, store it
        if hasattr(fn, "prompt_template"):
            pt = {}
            if hasattr(fn.prompt_template, "prompt_template_config"):
                ptc = {}
                if hasattr(fn.prompt_template.prompt_template_config, "template"):
                    ptc["template"] = fn.prompt_template.prompt_template_config.template
                pt["prompt_template_config"] = ptc
            fn_dict["prompt_template"] = pt

        return fn_dict

    data = {}

    # --- 1. Selection Strategy ---
    if multi_chat.selection_strategy and multi_chat.selection_strategy.function:
        data["selection_strategy"] = {
            "function": function_to_dict(multi_chat.selection_strategy.function)
        }
    else:
        data["selection_strategy"] = None

    # --- 2. Termination Strategy ---
    if multi_chat.termination_strategy and multi_chat.termination_strategy.function:
        data["termination_strategy"] = {
            "function": function_to_dict(multi_chat.termination_strategy.function)
        }
    else:
        data["termination_strategy"] = None

    # --- 3. Agents ---
    agents_list = []
    for agent in multi_chat.agents:
        agent_info = {
            "name": agent.name,
            "description": agent.description,
            "service_id": agent.service_id,
            "instructions": agent.instructions,
            "plugins": []
        }

        # --- 4. Plugins (if any) ---
        if agent.kernel.plugins:
            for _, plugin in agent.kernel.plugins.items():
                plugin_info = {
                    "name": plugin.name,
                    "description": plugin.description,
                    "functions": []
                }
                if plugin.functions:
                    for fn_obj in plugin.functions.values():
                        func_info = {
                            "name": fn_obj.name,
                            "fully_qualified_name": fn_obj.fully_qualified_name,
                            "description": fn_obj.description,
                        }
                        plugin_info["functions"].append(func_info)

                agent_info["plugins"].append(plugin_info)

        agents_list.append(agent_info)

    data["agents"] = agents_list

    # Convert to pretty JSON
    sk_calibrator_objects_json = json.dumps(data, indent=2)

    # Save to file if desired
    script_dir = os.path.dirname(__file__)  # Directory of the current script
    file_path = os.path.join(script_dir, "sk_calibrator_objects.json")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(sk_calibrator_objects_json)

    return sk_calibrator_objects_json


# This function is to modify a specific element in the multi_agent, by locating the element, and then modify the specific element to new value
def modify_multi_agent(multi_chat: AgentGroupChat, element_path: str, new_value):
    """
    Modify a specific element in the multi_chat by specifying a path string.
    The path can use dot notation to navigate attributes, and bracket notation for list indices.
    
    Examples of valid element_path strings:
        - "selection_strategy.function.name"
        - "selection_strategy.function.prompt"
        - "termination_strategy.function.prompt_template.prompt_template_config.template"
        - "agents[0].name"
        - "agents[0].plugins[1].functions[0].description"
    
    Args:
        multi_chat (AgentGroupChat): The multi-agent chat object you want to modify.
        element_path (str): A dot/bracket notation path to the attribute you want to modify.
        new_value: The new value to set for the specified attribute.
    """

    # 1. Convert bracket-based indexing into dot-based segments, so:
    #    "agents[0].name" --> "agents.0.name"
    #    "agents[0].plugins[1].functions[0].description" --> "agents.0.plugins.1.functions.0.description"
    #    This lets us consistently split on '.'
    path = element_path.replace('[', '.').replace(']', '')
    parts = path.split('.')  # e.g. "agents.0.name" -> ["agents", "0", "name"]

    # 2. Navigate down to the parent of the final attribute
    current_obj = multi_chat
    for p in parts[:-1]:
        if p.isdigit():
            # If the segment is purely digits, treat it as a list index
            current_obj = current_obj[int(p)]
        else:
            # Otherwise, treat it as an attribute name
            current_obj = getattr(current_obj, p)

    # 3. Modify the final attribute or list index
    final_part = parts[-1]
    if final_part.isdigit():
        # If the last part is digits, we assume current_obj is a list
        current_obj[int(final_part)] = new_value
    else:
        # Otherwise, set the attribute directly
        setattr(current_obj, final_part, new_value)


def read_variants_from_jsonl(file_path):
    """
    Read variants from a JSONL file and return them as a list of dictionaries.
    """
    variants = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            variants.append(json.loads(line))
    return variants


from sk_calibrator_component_assembler import AssembleAgentGroupChat      # NEW
from types import SimpleNamespace                                          # NEW

async def evaluate_all_variants(multi_chat,socketio):

    # Read variants from a JSONL file
    variants = read_variants_from_jsonl('sk_calibrator_experiment_1_variants.jsonl')
    print("Variants: ", variants)

    # Initial average score for each variants as a dictionary, to track the best variant
    average_scores = {i: 0.0 for i, variant in enumerate(variants)}

    # Modify the multi_agent based on the variants
    for i, variant in enumerate(variants):
        print("Variant: ", variant)
        socketio.emit('experiment_log', {'log': f"Evaluating variant {i+1}..."})
        score_total = 0.0
        score_count = 0

        # Build a brand‑new multi‑agent chat object from the variant instead
        # of patching the existing one.
        sk_components = SimpleNamespace(**variant)            # deserialize → attribute object
        # Each evaluation cycle will get a fresh AgentGroupChat
        # assembled exactly as described by the variant payload.

        azureopenai_endpoint = config.get("azureopenai_endpoint")
        test_questions = read_testcases_from_jsonl('./sample_sk_orchestrator_testcases.jsonl') 
        # Test case
        for question_01 in test_questions:
            user_input = question_01.question
            chat_history_input = convert_chat_history_to_sk_chat_history(question_01.chat_history)
            expected_answer = question_01.expected_answer

            # Re‑create a clean AgentGroupChat per question to avoid
            # cross‑pollinating history between test cases
            multi_chat_variant = AssembleAgentGroupChat(sk_components)   # NEW

            delta = ["planner_agent"]

            if True:
                question_2 = user_input 
                responses = []
                await multi_chat_variant.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=question_2))

                TERMINATION_KEYWORD = "TERMINATE"  # Define your termination keyword

                last_response = ""

                # Variable to accumulate all agent outputs.
                aggregated_outputs = ""

                try:

                    async for response in multi_chat_variant.invoke():
                        print("\n-------------------------------------\n", response.name, "\n",  response.content, "\n\n\n")
                        #socketio.emit('experiment_log', {'log': f"Multi-Agent {response.name}: {response.content}"})
                        last_response = response.content

                        # Append the response to the aggregated outputs.
                        aggregated_outputs += f"{response.name}: {response.content}\n"

                        if response.name in delta:
                            responses.append(response.content)
                            responses.append("*" * 50)

                        # Once the conversation ends, stop the loop
                        print("Is the conversation complete: ", multi_chat_variant.is_complete)
                        if multi_chat_variant.is_complete :
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

                system_answer = "\n".join(responses)
                
                # Calculate the similarity score between the system answer and the expected answer
                # For simplicity, let's assume a basic string match for this example
                # Similarity By GPT
                sim_prompt = f"Give a score between 0 and 1 for the following two answers. 1 means they are exactly the same, 0 means they are completely different.\n\nAnswer 1: {system_answer}\n\nAnswer 2: {expected_answer}\n\n Please only output the score value. Don't output anything else."
                messages = [{"role":"user","content": [{"type": "text", "text": sim_prompt}]}]
                MODEL_O1_MINI = "o1-mini-deploy"
                MODEL_VERSION_GPT_4O_MINI = "2025-01-01-preview"
                score = await call_aoai_withoutowndata_o1(messages, azureopenai_endpoint, MODEL_O1_MINI, MODEL_VERSION_GPT_4O_MINI)

                # Ensure the score is between 0 and 1, and convert to float
                score = float(score.strip())
                socketio.emit('experiment_log', {'log': f"Score for variant {i+1}: {score} \n system_answer: {system_answer} \n expected_answer: {expected_answer}"})
                score_total += score
                score_count += 1


        print("score_total: ", score_total)
        print("score_count: ", score_count)

        print("score: ", score_total / score_count)

        average_scores[i] = score_total / score_count


    # TODO: Now identify the best variant with highest score
    best_variant_key = max(average_scores, key=average_scores.get)
    best_variant_value = average_scores[best_variant_key]
    print(f"Best Variant: {best_variant_key} with score: {best_variant_value}")
    socketio.emit('experiment_log', {'log': f"Best Variant: {best_variant_key} with score: {best_variant_value}"})
    return best_variant_key, best_variant_value





#################USE CASE######################

if __name__ == "__main__":
    import asyncio

    multi_chat = load_group_chat()
    decode_multi_agent(multi_chat)
    multi_chat_json = convert_multi_agent_to_json(multi_chat)
    print(multi_chat_json)

    # TODO: Render the multi-agent as a web page

    # Print the existing instructions for sharepoint_knowledge_agent
    print("Before modification:\n", multi_chat.agents[0].instructions)
    # Modify the instructions for sharepoint_knowledge_agent
    new_instructions = """
    You are now updated with new instructions...
    """
    modify_multi_agent(multi_chat, "agents[0].instructions", new_instructions)
    # Confirm the instructions have been updated
    print("\nAfter modification:\n", multi_chat.agents[0].instructions)

    # Synchronously run the async evaluate_all_variants function
    asyncio.run(evaluate_all_variants(multi_chat))