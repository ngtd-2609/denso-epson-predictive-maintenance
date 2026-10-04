EPSON PERSON 1 - HANDOFF README
========================================================================

Release:
EPSON_PERSON1_HANDOFF_V1

Created:
2026-10-04T07:02:22.241327+07:00

CURRENT STATE
-------------
Producer-side technical readiness: PASS
Recipient acceptance: PENDING
Overall handoff state:
READY_FOR_RECIPIENT_ACCEPTANCE_WITH_OPEN_LIMITATIONS

DATA INTERFACE
--------------
Use:

    from loader_v1 import EpsonPreparedLoader

    loader = EpsonPreparedLoader()
    data = loader.load("k1", "train")

X:
[N, 3, 3000], float32, z-score

y_class:
0..5 configuration label

y_binary:
0 normal / 1 abnormal

recording_id:
source recording/file identifier

start_sample:
window start within the trimmed recording

DO NOT
------
- Do not refit scaler on validation or test.
- Do not z-score each window independently.
- Do not change split seed42 or scarcity seed101 silently.
- Do not add unused train faults into k1/k2/k4 experiments.
- Do not use test for model, preprocessing, threshold or
  hyperparameter selection.
- Do not treat windows as independent physical sessions.
- Do not claim session-independent evaluation.
- Do not claim multi-speed evaluation.
- Do not claim failure-time / remaining-life prediction.
- Do not claim this is DENSO factory measurement data.

PERSON 2
--------
Must independently confirm:
1. k1/train loads through loader_v1.
2. y_class and y_binary are understood separately.
3. Frozen scaler is used; no refit on validation/test.
4. Same real-record subset is retained across compared methods.
5. Metrics/aggregation respect recording/group identity.
6. Validation selects thresholds/aggregation/model choices.
7. Test is final held-out evaluation only.

PERSON 3
--------
Must independently confirm:
1. Only permitted train IDs for the selected k are used.
2. No checkpoint trained with a larger-k real pool is reused
   as if it belonged to a smaller-k scarcity experiment.
3. Generator output contains all X/Y/Z channels jointly.
4. Condition is configuration/fault class; this dataset does
   not provide a variable-speed research condition.
5. Generated values are in the correct z-score space and use
   the matching frozen scaler when inverse transforming.
6. Synthetic records carry synthetic provenance metadata:
   source_type, generator/version, seed and condition.
7. Synthetic data must not pretend to have a real source recording.

PERSON 4
--------
Use these evidence artifacts:
- dataset_card.txt
- source_register.csv
- access_and_scarcity.csv
- data_contract.json
- protocol_frozen.txt
- loader_selftest.json
- reproducibility_report.json
- HANDOFF_CHECKLIST.txt

OPEN LIMITATIONS
----------------
- Recording-session independence is unverified.
- Per-sample timestamps are unavailable; timestamp jitter was
  not measured.
- Dataset is one rig / nominal speed.
- Near-duplicate checks reduce risk but cannot prove absolute
  source independence.
- Dataset redistribution/use permissions remain to be verified
  for the intended external/public/commercial scope.
- Git commit history was unavailable at release time.
- Independent reproduction was performed in a new output
  directory on the same machine/environment, not cross-machine.
- Recipient acceptance has not yet been performed.

CHANGE CONTROL
--------------
Any change to source data, split, window length/stride,
scarcity membership, scaler or preprocessing requires a new
version and a new downstream-impact statement.
