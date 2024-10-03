import yaml

class YAMLParser():
    def __init__(self, path_base: str ="configs/", config: str = None) -> None:
        if config:
            ext = ".yaml"
            self.basename = path_base + config + ext
        else:
            raise Exception("Path is None")

        self.parse_config()
    
    def parse_config(self) -> dict:
         # Open the configuration file in read mode
        with open(self.basename, 'r') as f:
            # Read the data from the file and convert it from YAML format to Python objects
            config = yaml.safe_load(f)
        # Save the configuration to an object attribute
        return config

