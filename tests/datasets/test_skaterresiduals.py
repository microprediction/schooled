from schooled.datasets.skaterresiduals import concatenated_skater_residuals


def test_concatenated_skater_residuals():
    v = concatenated_skater_residuals(min_obs=500)
    assert len(v)>40000


if __name__=='__main__':
    test_concatenated_skater_residuals()