# Copyright (c) Microsoft. All rights reserved.

import importlib

class Calibrator_Component_Plugin:
    def __init__(self, plugin_class_name):

        self._Plugin_Class_Name = plugin_class_name #"my_module.MyPluginClass"
        self._Plugin_Name = ""
        self._Plugin_Description = ""

    def get_plugin(self):
        """
        Create and return an instance of the Calibrator Component Plugin,
        by dynamically loading the plugin class using its fully qualified name.
        """
        module_name, class_name = self._Plugin_Class_Name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        plugin_class = getattr(module, class_name)
        plugin = plugin_class()
        return plugin
    
    def get_plugin_class(self):
        """
        Create and return an instance of the Calibrator Component Plugin,
        by dynamically loading the plugin class using its fully qualified name.
        """
        module_name, class_name = self._Plugin_Class_Name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        plugin_class = getattr(module, class_name)
        return plugin_class