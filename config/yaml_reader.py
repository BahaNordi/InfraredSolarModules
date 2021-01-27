import yaml


def open_yaml(yaml_dir):
    with open(yaml_dir, 'r') as yml_config:
        out = yaml.load(yml_config)
    return out

