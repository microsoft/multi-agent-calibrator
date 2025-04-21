from flask import Flask, jsonify, abort, render_template, request
import json
import os
import asyncio
import time
import threading

# Import the configuration loader
from sk_calibrator_config import load_config
from sk_calibrator_object_loader import evaluate_all_variants, modify_multi_agent

# Load endpoints from YAML configuration
config = load_config("sample_sk_orchestrator_config.yaml")
azure_openai_endpoint = config.get("azureopenai_endpoint")
azure_openai_model = config.get("azureopenai_model")

from sample2_component_list import sk_component_abstraction, group_chat_info, agent_list, plugin_list, function_list, agent_topology, azure_openai_endpoint, azure_openai_model

# Load the multi-agent chat and convert it to JSON
from sk_calibrator_object_loader import load_group_chat, decode_multi_agent, convert_multi_agent_to_json
from sk_calibrator_component_assembler import AssembleAgentGroupChat

sk_components = sk_component_abstraction(group_chat_info, agent_list, plugin_list, function_list, agent_topology, azure_openai_endpoint, azure_openai_model)

# multi_chat = load_group_chat()
multi_chat = AssembleAgentGroupChat(
    sk_components
    #azure_openai_endpoint=azure_openai_endpoint, azure_openai_model="gpt-4o-mini-deploy"
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

# ------------------------------------------------------------------
# SocketIO – use the safe “threading” backend on Windows
# ------------------------------------------------------------------
socketio = SocketIO(app, async_mode="threading")   # <- was: socketio = SocketIO(app)


def convert_ui_variant_to_sk_objects(ui_variant: dict) -> dict:
    """
    Convert a UI variant (tree coming from the front‑end) into a
    ``sk_component_abstraction`` compatible dictionary.
    """

    # ------------------------------------------------------------------
    # 1.  Skeleton – copy the static parts that never change
    # ------------------------------------------------------------------
    result = {
        "group_chat_info":       group_chat_info,          # imported above
        "agent_list":            [],
        "plugin_list":           [],
        "function_list":         [],
        "agent_topology":        {"agents": []},
        "azure_openai_endpoint": azure_openai_endpoint,
        "azure_openai_model":    azure_openai_model,
    }

    # Sets used for de‑duplication
    _seen_plugins:   set[str] = set()
    _seen_functions: set[str] = set()

    # ------------------------------------------------------------------
    # 2.  Parse the UI tree
    # ------------------------------------------------------------------
    agents_ui = ui_variant.get("children", [])            # root node == "agents"
    for agent_node in agents_ui:
        # ---------------------- agent meta data -----------------------
        agent_meta = {
            "agent_name":        "",
            "agent_description": "",
            "agent_instruction": "",
            "service_id":        ""
        }
        plugins_for_topology: list[dict] = []

        for child in agent_node.get("children", []):
            n = child.get("name", "")
            if n.startswith("name:"):
                agent_meta["agent_name"] = child.get("value", "")
            elif n.startswith("description:"):
                agent_meta["agent_description"] = child.get("value", "")
            elif n.startswith("instructions"):
                agent_meta["agent_instruction"] = child.get("value", "")
            elif n.startswith("service_id"):
                agent_meta["service_id"] = child.get("value", "")
            elif n == "plugins":
                # ---------------- plugin section ----------------------
                for plugin_ui in child.get("children", []):
                    plugin_name = ""
                    functions_for_topology: list[dict] = []

                    for p_child in plugin_ui.get("children", []):
                        pn = p_child.get("name", "")
                        if pn.startswith("name:"):
                            plugin_name = p_child.get("value", "")
                        elif pn == "functions":
                            # ------------ functions -------------------
                            for func_ui in p_child.get("children", []):
                                fq_name = ""
                                func_desc = ""

                                for f_child in func_ui.get("children", []):
                                    fn = f_child.get("name", "")
                                    if fn.startswith("fully_qualified_name"):
                                        fq_name = f_child.get("value", "")
                                    elif fn.startswith("description"):
                                        func_desc = f_child.get("value", "")

                                # UI may omit fq_name; derive it if necessary
                                if not fq_name:
                                    local_name = func_ui.get("value", "")
                                    fq_name = f"{plugin_name}-{local_name}"

                                func_class = fq_name.replace("-", ".")
                                functions_for_topology.append(
                                    {"function_class_name": func_class}
                                )

                                if func_class not in _seen_functions:
                                    result["function_list"].append(
                                        {"function_class_name": func_class,
                                         "description": func_desc}
                                    )
                                    _seen_functions.add(func_class)

                    if plugin_name:
                        # ---- add to topology ----
                        plugins_for_topology.append(
                            {"plugin_name": plugin_name,
                             "functions": functions_for_topology}
                        )

                        # ---- global plugin list (dedup) ----
                        plugin_class = f"sample2_components.{plugin_name}"
                        if plugin_class not in _seen_plugins:
                            result["plugin_list"].append(
                                {"plugin_class_name": plugin_class,
                                 "plugin_name": plugin_name}
                            )
                            _seen_plugins.add(plugin_class)

        # ---------------------- finish agent --------------------------
        result["agent_list"].append(agent_meta)
        result["agent_topology"]["agents"].append(
            {"agent_name": agent_meta["agent_name"],
             "plugins": plugins_for_topology}
        )

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
    # Read directly from the sk calibrator objects (sk_component_abstraction object) json file format, so the server and client both use the same files. 
    sk_components_json = sk_components.to_dict()
    return sk_components_json

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

# ------------------------------------------------------------------
# helper that executes the async evaluator inside its own event‑loop
# ------------------------------------------------------------------
def _experiment_runner(multi_chat):
    """Run evaluate_all_variants in a fresh event‑loop and stream logs."""
    socketio.emit('experiment_log', {'log': "Starting experiment..."})
    socketio.sleep(0.5)

    socketio.emit('experiment_log', {'log': "Evaluating all variants..."})
    socketio.sleep(0.5)

    # create a private event‑loop for the async routine
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    best_key, best_val = loop.run_until_complete(
        evaluate_all_variants(multi_chat, socketio)      # <- pass socketio!
    )
    loop.close()

    socketio.emit('experiment_log', {'log': "Experiment complete."})
    socketio.emit('experiment_log',
                  {'log': f"Best variant: {best_key}, Value: {best_val}"})

# ------------------------------------------------------------------
# route – plain sync function that just launches the background task
# ------------------------------------------------------------------
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    variant_file = os.path.join(current_dir, 'sk_calibrator_experiment_1_variants.jsonl')
    if not os.path.exists(variant_file):
        abort(404, description="Variant file not found")

    # (still load/validate the file to give early feedback)
    try:
        with open(variant_file, 'r') as f:
            variants = [json.loads(line) for line in f]
    except Exception as e:
        abort(500, description=str(e))

    # kick off the long‑running task
    socketio.start_background_task(_experiment_runner, multi_chat)

    return jsonify({"status": "experiment_started"})

if __name__ == '__main__':
    socketio.run(app, debug=True)
