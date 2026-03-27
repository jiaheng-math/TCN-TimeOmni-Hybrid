PYTHON ?= python
TCN_UNC_CONFIG ?= configs/tcn/fd001_tcn_uncertainty_tuned.yaml
TCN_POINT_CONFIG ?= configs/tcn/fd001_tcn_point_tuned.yaml
HYBRID_UNC_CONFIG ?= configs/hybrid/fd001_hybrid_local.yaml
HYBRID_POINT_CONFIG ?= configs/hybrid/fd001_hybrid_point_local.yaml

.PHONY: install preprocess train train-point eval eval-point visualize visualize-point hybrid hybrid-point full full-point

install:
	$(PYTHON) -m pip install -r requirements.txt

preprocess:
	$(PYTHON) scripts/preprocess_tcn.py --config $(TCN_UNC_CONFIG)

train:
	$(PYTHON) scripts/train_tcn.py --config $(TCN_UNC_CONFIG)

train-point:
	$(PYTHON) scripts/train_tcn.py --config $(TCN_POINT_CONFIG)

eval:
	$(PYTHON) scripts/evaluate_tcn.py --config $(TCN_UNC_CONFIG)

eval-point:
	$(PYTHON) scripts/evaluate_tcn.py --config $(TCN_POINT_CONFIG)

visualize:
	$(PYTHON) scripts/visualize_tcn.py --config $(TCN_UNC_CONFIG)

visualize-point:
	$(PYTHON) scripts/visualize_tcn.py --config $(TCN_POINT_CONFIG)

hybrid:
	$(PYTHON) scripts/run_hybrid_demo.py --config $(HYBRID_UNC_CONFIG)

hybrid-point:
	$(PYTHON) scripts/run_hybrid_demo.py --config $(HYBRID_POINT_CONFIG)

full:
	$(PYTHON) scripts/run_full_pipeline.py --tcn-config $(TCN_UNC_CONFIG) --hybrid-config $(HYBRID_UNC_CONFIG)

full-point:
	$(PYTHON) scripts/run_full_pipeline.py --tcn-config $(TCN_POINT_CONFIG) --hybrid-config $(HYBRID_POINT_CONFIG)
