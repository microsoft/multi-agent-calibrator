# Copyright (c) Microsoft. All rights reserved.

import yaml
import os

def load_config(config_path=None):
    """
    Loads the YAML configuration file and returns a dictionary of configurations.
    If config_path is not provided, it defaults to 'sk_calibrator_config.yaml'
    located in the same directory as this module.
    """
    if config_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "sk_calibrator_config.yaml")
    
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return {}

if __name__ == "__main__":
    # For testing purposes
    config = load_config()
    print("Loaded configuration:", config)
