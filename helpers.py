import json


def load_json(path="config.json"):
    try:
        with open(path, 'r') as file:
            config_dict = json.load(file)
            return config_dict
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file at {path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
