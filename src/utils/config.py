def load_config(file_path):
    import json
    with open(file_path, 'r') as file:
        config = json.load(file)
    
    # Convert parameters to appropriate types
    config['volfrac'] = float(config['volfrac'])
    config['penalty'] = float(config['penalty'])
    config['x_min'] = float(config['x_min'])
    config['r_min'] = float(config['r_min'])
    config['Youngs_modulus'] = float(config['Youngs_modulus'])
    config['Poissons_ratio'] = float(config['Poissons_ratio'])
    
    return config

def save_config(file_path, config):
    import json
    with open(file_path, 'w') as file:
        json.dump(config, file, indent=4)

def get_parameter(config, key, default=None):
    return config.get(key, default)