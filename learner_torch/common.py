import datetime
import time
import warnings
from pathlib import Path
import inspect

from typing import List
import yaml

def load_yaml_config(args, role_type: str) -> None:
    if role_type not in {'actor', 'learner'}:
        raise ValueError('Invalid role type')

    # Load config file
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = None

    if config is not None and isinstance(config, dict):
        if role_type in config:
            for k, v in config[role_type].items():
                if k in args:
                    setattr(args, k, v)
                else:
                    warnings.warn(f"Invalid config item '{k}' ignored", RuntimeWarning)
        args.agent_config = config['agent'] if 'agent' in config else None
    else:
        args.agent_config = None


def save_yaml_config(config_path: Path, args, role_type: str, agent) -> None:
    class Dumper(yaml.Dumper):
        def increase_indent(self, flow=False, *_, **__):
            return super().increase_indent(flow=flow, indentless=False)

    if role_type not in {'actor', 'learner'}:
        raise ValueError('Invalid role type')

    with open(config_path, 'w') as f:
        args_config = {k: v for k, v in vars(args).items() if
                       not k.endswith('path') and k != 'agent_config' and k != 'config'}
        yaml.dump({role_type: args_config}, f, sort_keys=False, Dumper=Dumper)
        f.write('\n')
        yaml.dump({'agent': agent.export_config()}, f, sort_keys=False, Dumper=Dumper)


def create_experiment_dir(args, prefix: str) -> None:
    if args.exp_path is None:
        args.exp_path = prefix + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    args.exp_path = Path(args.exp_path)

    if args.exp_path.exists():
        raise FileExistsError(f'Experiment directory {str(args.exp_path)!r} already exists')

    args.exp_path.mkdir()


def get_config_params(obj_or_cls) -> List[str]:
    """
    Return configurable parameters in 'Agent.__init__' and 'Model.__init__' which appear after 'config'
    :param obj_or_cls: An instance of 'Agent' / 'Model' OR their corresponding classes (NOT base classes)
    :return: A list of configurable parameters
    """
    # import core  # Import inside function to avoid cyclic import

    # if inspect.isclass(obj_or_cls):
    #     if not issubclass(obj_or_cls, core.Agent) and not issubclass(obj_or_cls, core.Model):
    #         raise ValueError("Only accepts subclasses of 'Agent' or 'Model'")
    # else:
    #     if not isinstance(obj_or_cls, core.Agent) and not isinstance(obj_or_cls, core.Model):
    #         raise ValueError("Only accepts instances 'Agent' or 'Model'")

    sig = list(inspect.signature(obj_or_cls.__init__).parameters.keys())

    config_params = []
    config_part = False
    for param in sig:
        if param == 'config':
            # Following parameters should be what we want
            config_part = True
        elif param in {'args', 'kwargs'}:
            pass
        elif config_part:
            config_params.append(param)

    return config_params
