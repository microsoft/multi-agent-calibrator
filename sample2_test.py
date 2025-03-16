from sk_calibrator_component_assembler import AssembleAgentGroupChat
group_chat_info = {
    "group_chat_name": "Calibrator_Component_Chat",
    "group_chat_description": "Calibrator Component Chat",
    "group_chat_instruction": "You are a calibrator component agent. Your job is to assist the user with calibration tasks."
}
agent_list = [
    {"agent_name": "agent1", "agent_description": "", "agent_instruction": "", "service_id": ""},]
plugin_list = [
    {"plugin_class_name": "sample2_components.Scrum_Champ", "plugin_name": "Scrum_Champ"}]
function_list = [{"function_class_name": "sample2_components.search", "description": "description of func1"},]

AssembleAgentGroupChat(group_chat_info, agent_list,plugin_list, function_list, azure_openai_endpoint = "", azure_openai_model = "")