from pathlib import Path
import json
import numpy as np
from scipy.signal import correlate, correlation_lags

SEED = 20261004
FS = 3000.0
N = 30000
MAX_LAG = 6000
TRUE_LAG = 731

PERSON1 = Path(__file__).resolve().parent
QA_DIR = PERSON1 / "results" / "qa_v2"
OUT = QA_DIR / "near_duplicate_selftest.json"


def zscore_channels(x):
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)

    if np.any(sd <= 0):
        raise RuntimeError("Zero-variance channel in fixture")

    return (x - mu) / sd


def shift_common(x, lag):
    """
    Positive lag means y is delayed relative to x.

    Zero padding is used only for the synthetic fixture.
    """
    y = np.zeros_like(x)

    if lag == 0:
        y[:] = x
    elif lag > 0:
        y[lag:] = x[:-lag]
    else:
        k = -lag
        y[:-k] = x[k:]

    return y


def overlap_for_lag(x, y, lag):
    """
    Align x and y after estimating one COMMON lag for all 3 channels.

    Convention:
      positive lag => y is delayed relative to x
    """
    if lag == 0:
        return x, y

    if lag > 0:
        return x[:-lag], y[lag:]

    k = -lag
    return x[k:], y[:-k]


def estimate_common_lag(x, y, max_lag):
    """
    One joint lag only.

    Each channel is standardized separately for structural matching.
    Channel cross-correlations are averaged.
    """
    zx = zscore_channels(x)
    zy = zscore_channels(y)

    joint = None
    lags = None

    for c in range(3):
        cc = correlate(
            zy[:, c],
            zx[:, c],
            mode="full",
            method="fft",
        )

        lg = correlation_lags(
            len(zy[:, c]),
            len(zx[:, c]),
            mode="full",
        )

        denom = np.linalg.norm(zx[:, c]) * np.linalg.norm(zy[:, c])

        if denom <= 0:
            raise RuntimeError("Invalid normalization denominator")

        cc = cc / denom

        if joint is None:
            joint = cc
            lags = lg
        else:
            joint += cc

    joint = joint / 3.0

    mask = np.abs(lags) <= max_lag

    local_lags = lags[mask]
    local_joint = joint[mask]

    idx = int(np.nanargmax(local_joint))

    return int(local_lags[idx]), float(local_joint[idx])


def verify_pair(x, y):
    lag, lag_score = estimate_common_lag(x, y, MAX_LAG)

    xa, ya = overlap_for_lag(x, y, lag)

    zx = zscore_channels(xa)
    zy = zscore_channels(ya)

    per_channel_corr = []

    for c in range(3):
        r = np.corrcoef(zx[:, c], zy[:, c])[0, 1]
        per_channel_corr.append(float(r))

    residual = zx - zy

    joint_nrmse = float(np.sqrt(np.mean(residual ** 2)))
    joint_nmae = float(np.mean(np.abs(residual)))

    std_x = xa.std(axis=0)
    std_y = ya.std(axis=0)

    amplitude_ratio = std_y / std_x

    return {
        "estimated_common_lag": lag,
        "lag_score": lag_score,
        "matched_samples": int(len(xa)),
        "per_channel_corr": per_channel_corr,
        "mean_channel_corr": float(np.mean(per_channel_corr)),
        "joint_standardized_rmse": joint_nrmse,
        "joint_standardized_mae": joint_nmae,
        "std_ratio_y_over_x": amplitude_ratio.tolist(),
    }


def build_base_signal(rng):
    t = np.arange(N) / FS

    phases = np.array([0.15, 1.05, 2.10])

    x = np.zeros((N, 3), dtype=np.float64)

    for c in range(3):
        x[:, c] = (
            1.00 * np.sin(2*np.pi*20.0*t + phases[c])
            + 0.32 * np.sin(2*np.pi*40.0*t + 0.7*phases[c])
            + 0.16 * np.sin(2*np.pi*60.0*t + 1.3*phases[c])
            + 0.08 * np.sin(2*np.pi*117.0*t + 0.3*c)
        )

    # Common low-frequency modulation
    modulation = 1.0 + 0.06*np.sin(2*np.pi*0.73*t)

    x *= modulation[:, None]

    # Small nonperiodic content so identical-source copies
    # contain more than just the 20 Hz rotation signature.
    common_noise = rng.normal(0, 0.035, size=(N, 1))
    channel_noise = rng.normal(0, 0.020, size=(N, 3))

    x += common_noise
    x += channel_noise

    return x


def main():
    rng = np.random.default_rng(SEED)

    base = build_base_signal(rng)

    fixtures = {}

    fixtures["positive_exact_copy"] = base.copy()

    fixtures["positive_common_shift"] = shift_common(
        base,
        TRUE_LAG
    )

    fixtures["positive_shift_plus_noise"] = (
        shift_common(base, TRUE_LAG)
        + rng.normal(0, 0.015, size=base.shape)
    )

    fixtures["positive_scaled_shift_plus_noise"] = (
        1.18 * shift_common(base, TRUE_LAG)
        + rng.normal(0, 0.015, size=base.shape)
    )

    # Negative 1:
    # same nominal fundamental, but independently generated
    # modulation/noise/phases.
    t = np.arange(N) / FS

    neg1 = np.zeros((N, 3), dtype=np.float64)

    neg_phases = np.array([2.35, 0.41, 1.62])

    for c in range(3):
        neg1[:, c] = (
            1.00*np.sin(2*np.pi*20.0*t + neg_phases[c])
            + 0.21*np.sin(2*np.pi*40.0*t + 1.7*neg_phases[c])
            + 0.09*np.sin(2*np.pi*83.0*t + 0.8*c)
        )

    neg1 *= (
        1.0 + 0.09*np.sin(2*np.pi*0.41*t + 0.8)
    )[:, None]

    neg1 += rng.normal(0, 0.050, size=neg1.shape)

    fixtures["negative_independent_same_fundamental"] = neg1

    # Negative 2:
    # same 20 Hz family but deliberately different harmonic mixture,
    # per-channel phases and slow envelope.
    neg2 = np.zeros((N, 3), dtype=np.float64)

    for c in range(3):
        neg2[:, c] = (
            (0.65 + 0.13*c)
            * np.sin(2*np.pi*20.0*t + 0.9 + 0.8*c)
            + 0.42
            * np.sin(2*np.pi*37.0*t + 1.4*c)
            + 0.22
            * np.sin(2*np.pi*71.0*t + 0.4*c)
        )

    neg2 *= (
        1.0 + 0.11*np.sin(2*np.pi*0.29*t + 1.2)
    )[:, None]

    neg2 += rng.normal(0, 0.045, size=neg2.shape)

    fixtures["negative_same_frequency_different_source"] = neg2

    results = {}

    print("============================================================")
    print("NEAR-DUPLICATE SYNTHETIC SELF-TEST")
    print("============================================================")
    print("Seed:", SEED)
    print("Samples:", N)
    print("Known injected lag:", TRUE_LAG)
    print("Maximum searched lag:", MAX_LAG)
    print()

    for name, candidate in fixtures.items():
        r = verify_pair(base, candidate)
        results[name] = r

        print(name)
        print("  estimated lag        :", r["estimated_common_lag"])
        print("  lag score            :", f'{r["lag_score"]:.6f}')
        print(
            "  per-channel corr     :",
            [round(v, 6) for v in r["per_channel_corr"]]
        )
        print(
            "  mean channel corr    :",
            f'{r["mean_channel_corr"]:.6f}'
        )
        print(
            "  standardized NRMSE   :",
            f'{r["joint_standardized_rmse"]:.6f}'
        )
        print(
            "  standardized NMAE    :",
            f'{r["joint_standardized_mae"]:.6f}'
        )
        print(
            "  std ratio y/x        :",
            [round(v, 6) for v in r["std_ratio_y_over_x"]]
        )
        print()

    exact = results["positive_exact_copy"]
    shifted = results["positive_common_shift"]
    noisy = results["positive_shift_plus_noise"]
    scaled = results["positive_scaled_shift_plus_noise"]
    neg1r = results["negative_independent_same_fundamental"]
    neg2r = results["negative_same_frequency_different_source"]

    checks = {}

    checks["exact_copy_lag_zero"] = (
        exact["estimated_common_lag"] == 0
    )

    checks["exact_copy_corr_near_one"] = (
        exact["mean_channel_corr"] > 0.999
    )

    checks["shift_recovered"] = (
        abs(shifted["estimated_common_lag"] - TRUE_LAG) <= 1
    )

    checks["shift_corr_near_one"] = (
        shifted["mean_channel_corr"] > 0.995
    )

    checks["shift_plus_noise_recovered"] = (
        abs(noisy["estimated_common_lag"] - TRUE_LAG) <= 2
    )

    checks["shift_plus_noise_high_similarity"] = (
        noisy["mean_channel_corr"] > 0.95
    )

    checks["scaled_shift_recovered"] = (
        abs(scaled["estimated_common_lag"] - TRUE_LAG) <= 2
    )

    checks["scaled_shift_high_similarity"] = (
        scaled["mean_channel_corr"] > 0.95
    )

    # Negative controls must remain clearly separated from
    # known-source shifted copies. This is a fixture self-test criterion,
    # not a real-data auto-delete threshold.
    positive_floor = min(
        noisy["mean_channel_corr"],
        scaled["mean_channel_corr"],
    )

    negative_ceiling = max(
        neg1r["mean_channel_corr"],
        neg2r["mean_channel_corr"],
    )

    checks["negative_controls_separated"] = (
        positive_floor - negative_ceiling > 0.10
    )

    checks["negative_rmse_larger"] = (
        min(
            neg1r["joint_standardized_rmse"],
            neg2r["joint_standardized_rmse"],
        )
        >
        max(
            noisy["joint_standardized_rmse"],
            scaled["joint_standardized_rmse"],
        )
    )

    passed = all(checks.values())

    report = {
        "status": "PASS" if passed else "FAIL",
        "seed": SEED,
        "fs_hz": FS,
        "n_samples": N,
        "true_common_lag": TRUE_LAG,
        "max_lag": MAX_LAG,
        "important_note": (
            "Numerical criteria in this synthetic self-test validate "
            "the implementation on known fixtures only. They are not "
            "automatic duplicate/deletion thresholds for real Epson data."
        ),
        "checks": checks,
        "results": results,
    }

    QA_DIR.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("============================================================")
    print("SELFTEST STATUS:", report["status"])
    print("============================================================")

    for k, v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    print()
    print("Saved:", OUT)

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
