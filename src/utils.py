import yaml
from importlib import import_module


def read_yaml(path):
    """
    read .yaml file and return as dictionary
    :param path: path to .yaml file
    :return: parsed file as dictionary
    """
    with open(path, 'r') as file:
        try:
            parsed_yaml = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    file.close()

    return parsed_yaml

def create_instance(module_name, class_name, kwargs, *args):
    """
    create instance of a class
    :param module_name: str, module the class is in
    :param class_name: str, name of the class
    :param kwargs:
    :return: class instance
    """
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    if kwargs is None:
        instance = clazz(*args)
    else:
        instance = clazz(*args, **kwargs)
    return instance





if __name__ == '__main__':

    parsed_yaml = read_yaml('trainer/test/TEST.yaml')
    print(parsed_yaml)