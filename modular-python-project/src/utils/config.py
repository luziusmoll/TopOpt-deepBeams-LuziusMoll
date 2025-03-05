def load_config(file_path):
    import json
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def save_config(file_path, config):
    import json
    with open(file_path, 'w') as file:
        json.dump(config, file, indent=4)

def get_parameter(config, key, default=None):
    return config.get(key, default)