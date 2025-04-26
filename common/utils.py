import yaml


def load_yaml_file(file_path):
    """
    Loads YAML content from a file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: A dictionary representing the loaded YAML data, or None if an error occurs.
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None
