# Import the configuration loader
from sk_calibrator_config import load_config
# Load endpoints from YAML configuration
config = load_config("sample_sk_orchestrator_config.yaml")

azure_openai_endpoint = config.get("azureopenai_endpoint")
azure_openai_model = config.get("azureopenai_model")


group_chat_info = {
    "group_chat_name": "Calibrator_Component_Chat",
    "group_chat_description": "Calibrator Component Chat",
    "group_chat_instruction": "You are a calibrator component agent. Your job is to assist the user with calibration tasks.",
    "termination_function_prompt":"""You are a termination evaluator. Analyze the chat history and if the last agent response indicates that the task is complete, output the keyword 'TERMINATE'. Otherwise, output nothing.
        Chat History:
        {{$history}}
    """,
    "selection_function_prompt":"""You are the multi-agent coordinator. 
        Your task is to select exactly one agent for the next turn, based solely on the conversation history. 
        Select any one agent from the list of agents available i.e., 'agent1'.
        Rules:
        Output only the name of the selected agent. 
        Chat History: 
        {{$history}}
        Chat History:
        {{$history}}
    """
}
agent_list = [
    {"agent_name": "agent1", "agent_description": "Scrum Champ Agent", "agent_instruction": " TODO:  User Input: {user_input}  Chat History: {chat_history_input}  {{{{$history}}}}", "service_id": "agent1_service_id"},
    {"agent_name": "agent_pm", "agent_description": "Program Manager Agent", "agent_instruction": " TODO:  User Input: {user_input}  Chat History: {chat_history_input}  {{{{$history}}}}", "service_id": "agent1_service_id"}]
plugin_list = [
    {"plugin_class_name": "sample2_components.Scrum_Champ", "plugin_name": "Scrum_Champ"},
    {"plugin_class_name": "sample2_components.Program_Manager", "plugin_name": "Program_Manager"}]
function_list = [{"function_class_name": "sample2_components.search", "description": "description of func1"},
                 {"function_class_name": "sample2_components.fetch_team_capacity", "description": "Fetch the team capacity of current sprint"},]


agent_topology = {
    "agents": [
        {
            "agent_name": "agent1",
            "plugins": [
                {
                    "plugin_name": "Scrum_Champ",
                    "functions": [
                        {
                            "function_class_name": "sample2_components.search"
                        },
                        {
                            "function_class_name": "sample2_components.fetch_team_capacity"
                        }
                    ]
                }
            ]
        },
        {
            "agent_name": "agent_pm",
            "plugins": [
                {
                    "plugin_name": "Program_Manager",
                    "functions": [
                    ]
                }
            ]
        }
    ]
}