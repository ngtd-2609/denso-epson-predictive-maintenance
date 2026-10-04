EPSON LOADER V1
===============

Purpose
-------
Read the frozen prepared_v1 artifacts without recalculating
windows or normalization.

Main API
--------

from loader_v1 import EpsonPreparedLoader

loader = EpsonPreparedLoader()

train = loader.load("k1", "train")

print(train.X.shape)
print(train.y_class.shape)
print(train.recording_id[:5])
print(train.start_sample[:5])

Expected X layout:
[N, 3, 3000]

Profiles:
- normal_only
- k1
- k2
- k4

Splits:
- train
- validation
- test

Aliases:
- profile "1" -> "k1"
- profile "2" -> "k2"
- profile "4" -> "k4"
- split "val" -> "validation"

PyTorch adapter:
dataset = loader.to_torch_dataset(train, target="class")

Important
---------
This loader does not:
- fit a scaler
- filter signals
- resample signals
- re-window raw CSV
- alter labels
- alter split membership
- select models or thresholds

The test split may be structurally loaded for frozen
pre-model integrity checks, but must not guide model,
threshold, preprocessing or hyperparameter decisions.
