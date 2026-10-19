"""Build only the selected training or preprocessing experiment."""
from importlib import import_module

exp_registry = {
    "exp_video": (".exp_video", "VideoPredictionExperiment"),
    "exp_inference": (".exp_video", "VideoPredictionExperiment"),
    "exp_idm": (".exp_idm", "IDMTrainingExperiment"),
    "process_data": (".process_data", "ProcessDataExperiment"),
}


def build_experiment(cfg, output_dir=None, ckpt_path=None):
    name = cfg.experiment._name
    if name not in exp_registry:
        raise ValueError(f"Unknown experiment {name!r}; choose from {list(exp_registry)}")
    module, class_name = exp_registry[name]
    experiment = getattr(import_module(module, __name__), class_name)
    return experiment(cfg, output_dir, ckpt_path)
