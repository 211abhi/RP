import numpy as np
from mimic_forecaster.data.preprocessing import causal_impute, fit_robust_stats


def test_causal_fill_is_bounded_and_preserves_mask():
    values=np.array([[1.0],[np.nan],[np.nan],[np.nan]],dtype=np.float32); mask=np.array([[1],[0],[0],[0]],dtype=np.float32)
    filled,age=causal_impute(values,mask,np.array([9.0],dtype=np.float32),max_gap=2)
    assert filled[:,0].tolist()==[1.0,1.0,1.0,9.0]
    assert mask[:,0].tolist()==[1.0,0.0,0.0,0.0]
    assert age[-1,0]==1.0


def test_stats_ignore_unobserved_values():
    values=np.array([[1.0],[2.0],[999.0]],dtype=np.float32); mask=np.array([[1],[1],[0]],dtype=np.float32)
    stats=fit_robust_stats(values,mask)
    assert stats.median[0]==1.5
    assert stats.scale[0]==0.5

