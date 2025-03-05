def read_json_file(file_path):
    import json
    with open(file_path, 'r') as file:
        return json.load(file)

def write_json_file(file_path, data):
    import json
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_text_file(file_path, data):
    with open(file_path, 'w') as file:
        file.write(data)