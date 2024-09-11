import os
import yaml


def yaml_config_hook(config_file): #config_file='config/config.yaml'
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f: #<_io.TextIOWrapper name='config/config.yaml' mode='r' encoding='UTF-8'>
        cfg = yaml.safe_load(f) #cfg={dict:18}
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg
