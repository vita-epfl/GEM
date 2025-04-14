from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "gem.modules.GeneralConditioner",
    "params": {"emb_models": list()},
}
