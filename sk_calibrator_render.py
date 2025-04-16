# Copyright (c) Microsoft. All rights reserved.
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
multi_chat = AssembleAgentGroupChat(group_chat_info, agent_list, plugin_list, function_list, agent_topology, azure_openai_endpoint=azure_openai_endpoint, azure_openai_model="gpt-4o-mini-deploy")
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

@app.route('/')
def index():
    # Pass the multi-agent component lists to the template.
    return render_template('sk_calibrator_render.html',
                           agent_list=agent_list,
                           plugin_list=plugin_list,
                           function_list=function_list)

@app.route('/get_tree', methods=['GET'])
def get_tree():
    # Build the full file path relative to the current directory
    file_path = os.path.join(current_dir, 'sk_calibrator_objects.json')
    
    if not os.path.exists(file_path):
        abort(404, description="JSON file not found")
    
    try:
        with open(file_path, 'r') as f:
            tree = json.load(f)
    except Exception as e:
        abort(500, description=str(e))
        
    return jsonify(tree)

@app.route('/save_variant', methods=['POST'])
def save_variant():
    data = request.get_json()
    if not data or 'modified_tree' not in data:
        abort(400, description="Invalid data. Expecting 'modified_tree'.")
    variant = data["modified_tree"]
    variant_file = os.path.join(current_dir, 'sk_calibrator_experiment_1_variants.jsonl')
    try:
        with open(variant_file, 'a') as f:
            f.write(json.dumps(variant) + "\n")
    except Exception as e:
        abort(500, description=str(e))
    return jsonify({"status": "success"})

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
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
    time.sleep(0.5)
    socketio.emit('experiment_log', {'log': "Evaluating all variants..."})
    time.sleep(0.5)

    # Run the experiment with each variant
    # Synchronously run the async evaluate_all_variants function
    best_variant_key, best_variant_value = asyncio.run(evaluate_all_variants(multi_chat,socketio))
    
    socketio.emit('experiment_log', {'log': "Experiment complete."})
    return jsonify({"result": f"Best variant: {best_variant_key}, Value: {best_variant_value}"})

if __name__ == '__main__':
    socketio.run(app, debug=True)