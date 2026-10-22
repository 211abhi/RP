import pytest
from mimic_forecaster.config import validate_config


def test_invalid_split_ratios_fail():
    cfg={"project":{},"data":{"split_ratios":{"train":.8,"val":.2,"test":.2}},"task":{"context_hours":24,"horizon_hours":1},"tokenizer":{"patch_size":4},"encoder":{},"training":{}}
    with pytest.raises(ValueError,match="sum to 1"): validate_config(cfg)

