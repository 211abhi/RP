"""The two new heads: medication labels and withheld-column imputation."""
import numpy as np
import pytest
import torch
from datetime import datetime, timedelta

from mimic_forecaster.data.mimic import HourlyStay
from mimic_forecaster.data.preprocessing import RobustStats
from mimic_forecaster.data.windows import build_windows, drug_prevalence
from mimic_forecaster.training import apply_column_dropout, drop_columns, masked_bce, multitask_loss

RATIOS = {"train": 1.0, "val": 0.0, "test": 0.0}


def _stay(hours, drug_active=None, n_drugs=2):
    times = [datetime(2020, 1, 1) + timedelta(hours=h) for h in range(hours)]
    return HourlyStay("1", "10", times, ("x",), [[float(h)] for h in range(hours)],
                      [[1] for _ in range(hours)], [1], "900", ("a", "b"),
                      drug_active if drug_active is not None else [], [1] * n_drugs if drug_active else [])


def _windows(stay, n_drugs=2, context=2, horizon=1):
    stats = RobustStats(np.array([0.], np.float32), np.array([1.], np.float32))
    return build_windows([stay], stats, context=context, horizon=horizon, stride=1, max_gap=1,
                         seed=42, ratios=RATIOS, n_drugs=n_drugs)


def test_drug_label_is_any_exposure_inside_the_horizon():
    # drug a active only at hour 3, drug b never. Horizon 1, context 2 -> windows target 2,3,4.
    active = [[0, 0], [0, 0], [0, 0], [1, 0], [0, 0]]
    result = _windows(_stay(5, active))
    assert result.drug_targets.shape == (3, 2)
    assert result.drug_targets[:, 0].tolist() == [0.0, 1.0, 0.0]   # window ending at hour 3
    assert result.drug_targets[:, 1].tolist() == [0.0, 0.0, 0.0]
    assert result.drug_mask.min() == 1.0


def test_a_stay_without_prescriptions_is_masked_out_not_labelled_negative():
    """No join means no supervision. Zero labels with a zero mask, never zero labels alone."""
    result = _windows(_stay(5, drug_active=None))
    assert result.drug_mask.sum() == 0.0
    assert result.drug_targets.sum() == 0.0
    # A masked class must contribute exactly nothing to the loss.
    logits = torch.full((3, 2), 5.0, requires_grad=True)
    loss = masked_bce(logits, torch.from_numpy(result.drug_targets), torch.from_numpy(result.drug_mask))
    assert float(loss.detach()) == 0.0


def test_drug_prevalence_divides_by_the_mask_not_the_batch():
    active = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
    prevalence = drug_prevalence(_windows(_stay(5, active)))
    assert prevalence.tolist() == [1.0, 0.0]


def test_column_dropout_never_hides_every_column():
    present = torch.ones(64, 4)
    keep = drop_columns(present, rate=0.99)
    assert keep.sum(1).min() >= 1.0
    assert keep.max() <= 1.0


def test_column_dropout_cannot_keep_a_column_the_source_lacks():
    present = torch.tensor([[1.0, 0.0, 1.0]])
    assert drop_columns(present, rate=0.0).tolist() == [[1.0, 0.0, 1.0]]
    for _ in range(20):
        keep = drop_columns(present, rate=0.5)
        assert keep[0, 1] == 0.0


def test_hidden_columns_leak_through_no_channel():
    """A hidden column must be indistinguishable from one the source never had."""
    values = torch.full((1, 4, 3), 7.0); observed = torch.ones(1, 4, 3)
    present = torch.ones(1, 3); age = torch.zeros(1, 4, 3)
    keep = torch.tensor([[1.0, 0.0, 1.0]])
    v, o, p, a = apply_column_dropout(values, observed, present, age, keep)
    assert v[0, :, 1].tolist() == [0.0] * 4      # scaled median
    assert o[0, :, 1].tolist() == [0.0] * 4      # never observed
    assert p[0, 1] == 0.0                        # column absent
    assert a[0, :, 1].tolist() == [1.0] * 4      # maximally stale
    assert v[0, :, 0].tolist() == [7.0] * 4      # kept columns untouched


def test_impute_loss_only_scores_hidden_and_observed_cells():
    values = torch.zeros(1, 2, 2)
    observed = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
    batch = (values, observed, torch.ones(1, 2), torch.zeros(1, 2, 2),
             torch.zeros(1, 1, 2), torch.ones(1, 1, 2), torch.zeros(1, 0), torch.zeros(1, 0))
    weights = {"forecast": 1.0, "impute": 1.0}
    output = {"forecast": torch.zeros(1, 1, 2), "impute": torch.tensor([[3.0, 3.0]])}
    # Column 0 kept -> not scored. Column 1 hidden but unobserved at the last hour -> not scored.
    _, parts = multitask_loss(output, batch, weights, None, keep=torch.tensor([[1.0, 0.0]]))
    assert parts["impute"] == 0.0
    # Now hide column 0, which IS observed at the last hour: error 3.0 -> squared 9.0.
    _, parts = multitask_loss(output, batch, weights, None, keep=torch.tensor([[0.0, 1.0]]))
    assert parts["impute"] == pytest.approx(9.0)


def test_new_start_label_ignores_running_drugs_and_unscores_a_short_future():
    # a: off, then on at hour 3. b: on throughout (already running, never a "new start").
    active = [[0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1]]
    stats = RobustStats(np.array([0.], np.float32), np.array([1.], np.float32))
    w = build_windows([_stay(6, active)], stats, context=2, horizon=1, stride=1, max_gap=1,
                      seed=42, ratios=RATIOS, n_drugs=2, start_horizon=2)
    # Last observed hours 1,2,3,4. Hour 3: a is already on -> unscored. Hour 4: the 2h future
    # runs past the stay -> unscored, not negative.
    assert w.start_targets[:, 0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert w.start_mask[:, 0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert w.start_mask[:, 1].sum() == 0.0                       # b is running, so never scored
    assert w.drug_history.shape == (4, 2, 2)
