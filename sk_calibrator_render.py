from flask import Flask, jsonify, abort, render_template, request
import json
import os
import asyncio
import time

# Import the configuration loader
from sk_calibrator_config import load_config
from sk_calibrator_object_loader import evaluate_all_variants, modify_multi_agent

# Load endpoints from YAML configuration
config = load_config("sample_sk_orchestrator_config.yaml")
azure_openai_endpoint = config.get("azureopenai_endpoint")
azure_openai_model = config.get("azureopenai_model")

from sample2_component_list import agent_list, plugin_list, function_list, group_chat_info, agent_topology, azure_openai_endpoint, azure_openai_model

# Load the multi-agent chat and convert it to JSON
from sk_calibrator_object_loader import load_group_chat, decode_multi_agent, convert_multi_agent_to_json
from sk_calibrator_component_assembler import AssembleAgentGroupChat

# multi_chat = load_group_chat()
multi_chat = AssembleAgentGroupChat(
    group_chat_info, agent_list, plugin_list, function_list, agent_topology,
    azure_openai_endpoint=azure_openai_endpoint, azure_openai_model="gpt-4o-mini-deploy"
)
decode_multi_agent(multi_chat)
multi_chat_json = convert_multi_agent_to_json(multi_chat)
print(multi_chat_json)

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Serve templates from the current directory.
app = Flask(__name__, template_folder=current_dir)

# Import SocketIO and initialize it
from flask_socketio import SocketIO, emit
socketio = SocketIO(app)


def convert_ui_variant_to_sk_objects(ui_variant: dict) -> dict:
    """
    Convert a UI variant selection into the sk_calibrator_objects format, 
    focusing on reconstructing the agent_topology with exact function_class_name matches.
    """
    result = {}
    # Initialize agent_topology with an empty plugin list
    agent_topology = {"plugin_list": []}
    
    # Determine the list of plugin nodes in the UI variant (could be stored under 'children' or 'plugin_list')
    plugins_ui = ui_variant.get("children") or ui_variant.get("plugin_list") or ui_variant.get("plugins") or []
    
    for plugin_node in plugins_ui:
        # Plugin’s fully‑qualified class name comes directly from the UI node
        plugin_class_name = plugin_node.get("fully_qualified_name", "")
        
        plugin_object = {
            "plugin_class_name": plugin_class_name,
            "function_list": []
        }
        
        # Get function nodes under this plugin in the UI (could be under 'children' or 'function_list' keys)
        function_nodes = plugin_node.get("children") or plugin_node.get("function_list") or plugin_node.get("functions") or []
        for func_node in function_nodes:
            ui_full_name = func_node.get("fully_qualified_name", "")
            # Extract the function_name from UI's "PluginName-function_name"
            if "-" in ui_full_name:
                # Split into plugin part and function part (maxsplit=1 to handle only the first hyphen)
                _, function_name = ui_full_name.split("-", 1)
            else:
                # If no hyphen, treat the whole name as the function name (perhaps already fully qualified in some cases)
                function_name = ui_full_name
            
            # Directly derive fully‑qualified function class name from the UI entry.
            # If the UI encodes it as "PluginName-function_name", convert the hyphen to a dot.
            full_func_class = ui_full_name.replace("-", ".")
            plugin_object["function_list"].append({"function_class_name": full_func_class})
        
        # Add the plugin object to the agent_topology's plugin list (even if its function_list is empty)
        agent_topology["plugin_list"].append(plugin_object)
    
    # Even if plugins_ui is empty (no plugins selected), agent_topology with an empty plugin_list is retained
    result["agent_topology"] = agent_topology
    return result


@app.route('/')
def index():
    # Pass the multi-agent component lists to the template.
    return render_template(
        'sk_calibrator_render.html',
        agent_list=agent_list,
        plugin_list=plugin_list,
        function_list=function_list
    )

@app.route('/get_tree', methods=['GET'])
def get_tree():
    # Build the full file path relative to the current directory
    # Convert the sk_component_abstraction to sk_calibrator_objects format on the fly. 
    # The sk_calibrator_objects format is only to be used to communicate with the UI.  Not to use in the backend.

    multi_chat_json = convert_multi_agent_to_json(multi_chat)
    print(multi_chat_json)

    #file_path = os.path.join(current_dir, 'sk_calibrator_objects.json')
    #if not os.path.exists(file_path):
    #    abort(404, description="JSON file not found")
    #try:
    #    with open(file_path, 'r') as f:
    #        tree = json.load(f)
    #except Exception as e:
    #    abort(500, description=str(e))
    #return jsonify(tree)
    #return  #jsonify(multi_chat_json)
    return  multi_chat_json

@app.route('/save_variant', methods=['POST'])
def save_variant():
    data = request.get_json()
    if not data or 'modified_tree' not in data:
        abort(400, description="Invalid data. Expecting 'changes'.")
    variant = data["modified_tree"]
    # Convert the variant to sk_calibrator_objects format on the fly, before save to sk_calibrator_experiment_1_variants.jsonl
    ui_variant = data["modified_tree"]
    sk_variant = convert_ui_variant_to_sk_objects(ui_variant)
    variant_file = os.path.join(current_dir, 'sk_calibrator_experiment_1_variants.jsonl')
    try:
        with open(variant_file, 'a') as f:
            f.write(json.dumps(sk_variant) + "\n")
    except Exception as e:
        abort(500, description=str(e))
    return jsonify({"status": "success"})

@app.route('/run_experiment', methods=['POST'])
async def run_experiment():
    # Read from experiment with a list of variants
    variant_file = os.path.join(current_dir, 'sk_calibrator_experiment_1_variants.jsonl')
    if not os.path.exists(variant_file):
        abort(404, description="Variant file not found")
    try:
        with open(variant_file, 'r') as f:
            variants = [json.loads(line) for line in f]
    except Exception as e:
        abort(500, description=str(e))
    # Emit log messages in real time to UI
    socketio.emit('experiment_log', {'log': "Starting experiment..."})
    await socketio.sleep(0.5)
    socketio.emit('experiment_log', {'log': "Evaluating all variants..."})
    await socketio.sleep(0.5)
    # Run the experiment with each variant
    # Synchronously run the async evaluate_all_variants function
    best_variant_key, best_variant_value = await evaluate_all_variants(multi_chat)
    socketio.emit('experiment_log', {'log': "Experiment complete."})
    return jsonify({"result": f"Best variant: {best_variant_key}, Value: {best_variant_value}"})

if __name__ == '__main__':
    socketio.run(app, debug=True)
