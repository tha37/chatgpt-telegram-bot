import json
import importlib
import inspect
import os
import pkgutil

from plugins.plugin import Plugin


class PluginManager:
    """
    A class to manage the plugins and call the correct functions
    """

    def __init__(self, config):
        enabled_plugins = [p for p in config.get('plugins', []) if p]

        alias = {
            'wolfram': 'wolfram_alpha',
            'deepl_translate': 'deepl',
            'whois': 'whois_',
        }

        plugin_dir = os.path.join(os.path.dirname(__file__), 'plugins')
        available = {}
        for _, module_name, _ in pkgutil.iter_modules([plugin_dir]):
            if module_name == 'plugin':
                continue
            module = importlib.import_module(f'plugins.{module_name}')
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Plugin) and obj is not Plugin:
                    plugin_name = next((k for k, v in alias.items() if v == module_name), module_name)
                    available[plugin_name] = obj
                    break

        self.plugins = [available[name]() for name in enabled_plugins if name in available]

    def get_functions_specs(self):
        """
        Return the list of function specs that can be called by the model
        """
        return [spec for specs in map(lambda plugin: plugin.get_spec(), self.plugins) for spec in specs]

    async def call_function(self, function_name, helper, arguments):
        """
        Call a function based on the name and parameters provided
        """
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return json.dumps({'error': f'Function {function_name} not found'})
        return json.dumps(await plugin.execute(function_name, helper, **json.loads(arguments)), default=str)

    def get_plugin_source_name(self, function_name) -> str:
        """
        Return the source name of the plugin
        """
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return ''
        return plugin.get_source_name()

    def __get_plugin_by_function_name(self, function_name):
        return next((plugin for plugin in self.plugins
                    if function_name in map(lambda spec: spec.get('name'), plugin.get_spec())), None)
