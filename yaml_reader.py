import yaml

class YAMLParser():
    def __init__(self, path_base: str ="configs/", config: str = None):
        if path_base and config is not None:
            self.basename = path_base + config 
        else:
            raise Exception("Path is None")

        self.parse_config()
    
    def parse_config(self):
         # Open the configuration file in read mode
        with open(self.basename, 'r') as f:
            # Read the data from the file and convert it from YAML format to Python objects
            config = yaml.safe_load(f)
        # Save the configuration to an object attribute
        self.config = config
