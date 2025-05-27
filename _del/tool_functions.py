from copy import deepcopy
from importlib import import_module


def init_instance_by_config(config: dict, default_class_dir: str = "rm_gallery", **kwargs):
    config_copy = deepcopy(config)
    origin_class_path: str = config_copy.pop("class")
    if not origin_class_path:
        raise RuntimeError("empty class path!")

    class_path_list = []
    if default_class_dir and not origin_class_path.startswith(default_class_dir):
        class_path_list.append(default_class_dir)

    class_path_split = origin_class_path.split(".")
    class_file_name: str = class_path_split[-1]

    class_path_list.extend(class_path_split[-1:])

    module = import_module(".".join(class_path_list))
    config_copy.update(kwargs)
    return getattr(module, class_file_name)(**config_copy)