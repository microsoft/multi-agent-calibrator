# Import the configuration loader
from sk_calibrator_config import load_config
# Load endpoints from YAML configuration
config = load_config("sample_sk_orchestrator_config.yaml")

azure_openai_endpoint = config.get("azureopenai_endpoint")
azure_openai_model = config.get("azureopenai_model")

group_chat_info = {
    "group_chat_name": "Calibrator_Component_Chat",
    "group_chat_description": "Calibrator Component Chat",
    "group_chat_instruction": "You are a calibrator component agent. Your job is to assist the user with calibration tasks."
}
agent_list = [
    {"agent_name": "agent1", "agent_description": "TODO", "agent_instruction": " TODO:  User Input: {user_input}  Chat History: {chat_history_input}  {{{{$history}}}}", "service_id": "agent1_service_id"},]
plugin_list = [
    {"plugin_class_name": "sample2_components.Scrum_Champ", "plugin_name": "Scrum_Champ"}]
function_list = [{"function_class_name": "sample2_components.search", "description": "description of func1"},]
