from datetime import datetime,timedelta
import numpy as np
from mimic_forecaster.data.mimic import HourlyStay
from mimic_forecaster.data.preprocessing import RobustStats
from mimic_forecaster.data.windows import build_windows,subject_split


def test_subject_split_is_stable():
    assert subject_split("123",42,.7,.15)==subject_split("123",42,.7,.15)


def test_windows_skip_unobserved_targets():
    times=[datetime(2020,1,1)+timedelta(hours=i) for i in range(5)]
    stay=HourlyStay("1","10",times,("x",),[[1.],[2.],[3.],[4.],[5.]],[[1],[1],[1],[0],[1]],[1])
    stats=RobustStats(np.array([0.],dtype=np.float32),np.array([1.],dtype=np.float32))
    result=build_windows([stay],stats,context=2,horizon=1,stride=1,max_gap=1,seed=42,ratios={"train":.7,"val":.15,"test":.15})
    assert len(result.values)==2
    assert result.values.shape==(2,2,1)

