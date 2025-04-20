# Import the configuration loader
from sk_calibrator_config import load_config
import collections.abc
import json
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

def _to_serializable(obj):
    """
    Recursively convert an arbitrary object into JSON‑serializable
    primitives.  Custom classes are flattened into a dict of their
    public (non‑callable) attributes.
    """
    # Primitives
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Mappings
    if isinstance(obj, collections.abc.Mapping):
        return {k: _to_serializable(v) for k, v in obj.items()}

    # Iterables
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]

    # Custom objects with __dict__
    if hasattr(obj, "__dict__"):
        return {
            k: _to_serializable(v)
            for k, v in vars(obj).items()
            if not k.startswith("_") and not callable(v)
        }

    # Fallback
    return str(obj)

class sk_component_abstraction:
    def __init__(self, group_chat_info, agent_list, plugin_list, function_list, agent_topology, azure_openai_endpoint=None, azure_openai_model=None):
        self.group_chat_info = group_chat_info
        self.agent_list = agent_list
        self.plugin_list = plugin_list
        self.function_list = function_list
        self.agent_topology = agent_topology
        self.azure_openai_endpoint = azure_openai_endpoint
        self.azure_openai_model = azure_openai_model

    # ---- convenience helpers ----
    def to_dict(self):
        """Return a fully JSON‑serializable dict representation."""
        return _to_serializable(self)

    def __repr__(self):
        """Pretty JSON string shown in interactive prints / logs."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
