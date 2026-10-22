import torch
from mimic_forecaster.models.encoder import TimeSeriesEncoder
from mimic_forecaster.models.forecaster import MultiTaskForecaster
from mimic_forecaster.models.tokenizer import MaskAwarePatchTokenizer


def _model(n_features=3,n_drugs=5,horizon=1):
    return MultiTaskForecaster(MaskAwarePatchTokenizer(n_features,2,16,include_age=True),
                               TimeSeriesEncoder(16,4,1,32,0.0,3),16,horizon,n_features,n_drugs)


def _inputs(batch=2,time=6,features=3):
    values=torch.zeros(batch,time,features)
    return values,torch.ones_like(values),torch.ones(batch,features),torch.zeros_like(values)


def test_every_head_has_the_contracted_shape():
    out=_model()(*_inputs())
    assert out["forecast"].shape==(2,1,3)
    assert out["impute"].shape==(2,3)
    assert out["drug_logits"].shape==(2,5)


def test_drug_head_is_absent_when_no_drugs_are_configured():
    out=_model(n_drugs=0)(*_inputs())
    assert "drug_logits" not in out
    assert set(out)=={"forecast","impute"}
