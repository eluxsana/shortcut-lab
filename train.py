"""A deliberately small, reproducible shortcut-learning experiment. Standard library only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def dataset(n, alignment, seed):
    """Keep the same latent examples across alignment levels (common random numbers)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        y = rng.choice((-1, 1))
        stable = y + rng.gauss(0, 1)
        u, noise = rng.random(), rng.gauss(0, 0.15)
        aligned = u < alignment
        shortcut = (y if aligned else -y) + noise
        rows.append((stable, shortcut, (y + 1) // 2, aligned))
    return rows


def loss_gradient(rows, weights, stable_only=False, penalty=0.01):
    loss, g0, g1, gb = 0.0, 0.0, 0.0, 0.0
    for stable, shortcut, y, _ in rows:
        shortcut = 0.0 if stable_only else shortcut
        z = weights[0] * stable + weights[1] * shortcut + weights[2]
        loss += max(z, 0) - y * z + math.log1p(math.exp(-abs(z)))
        residual = sigmoid(z) - y
        g0 += residual * stable
        g1 += residual * shortcut
        gb += residual
    n = len(rows)
    return (
        loss / n + penalty * (weights[0] ** 2 + weights[1] ** 2) / 2,
        (g0 / n + penalty * weights[0], g1 / n + penalty * weights[1], gb / n),
    )


def train(rows, stable_only=False, epochs=240, learning_rate=0.25):
    weights = [0.0, 0.0, 0.0]
    losses = []
    for step in range(epochs):
        loss, gradient = loss_gradient(rows, weights, stable_only)
        if step % 10 == 0:
            losses.append({"step": step, "loss": loss})
        weights = [w - learning_rate * g for w, g in zip(weights, gradient)]
    return {"weights": weights, "losses": losses, "stable_only": stable_only}


def wilson(correct, n):
    if n == 0:
        return None
    z = 1.959963984540054
    p, denom = correct / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [center - half, center + half]


def evaluate(model, rows):
    cells = {name: [0, 0] for name in ("overall", "aligned", "conflicting")}
    for stable, shortcut, y, aligned in rows:
        w = model["weights"]
        z = w[0] * stable + (0 if model["stable_only"] else w[1] * shortcut) + w[2]
        correct = int(int(z >= 0) == y)
        for name in ("overall", "aligned" if aligned else "conflicting"):
            cells[name][0] += correct
            cells[name][1] += 1
    return {
        name: {
            "correct": c,
            "n": n,
            "accuracy": c / n if n else None,
            "wilson95": wilson(c, n),
        }
        for name, (c, n) in cells.items()
    }


def experiment(n_train=3000, n_test=4000):
    aligned_train = dataset(n_train, 0.95, 17)
    balanced_train = dataset(n_train, 0.5, 17)
    specifications = [
        ("erm", "Shortcut-rich training", aligned_train, False),
        ("balanced", "Balanced training", balanced_train, False),
        ("stable", "Stable feature only", aligned_train, True),
    ]
    models = []
    for key, label, rows, stable_only in specifications:
        model = train(rows, stable_only)
        model.update({"key": key, "label": label, "train": evaluate(model, rows)})
        model["curve"] = [
            {"alignment": i / 20, **evaluate(model, dataset(n_test, i / 20, 9281))}
            for i in range(21)
        ]
        models.append(model)
    return {
        "schema": 1,
        "data": "Synthetic binary classification; no employer data or pretrained model.",
        "train_seed": 17,
        "test_seed": 9281,
        "n_train": n_train,
        "n_test": n_test,
        "epochs": 240,
        "learning_rate": 0.25,
        "l2": 0.01,
        "models": models,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("report"))
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing report"
    )
    args = parser.parse_args()
    if not args.force and any(
        (args.out / name).exists() for name in ("results.json", "index.html")
    ):
        parser.error("Report exists. Choose a new --out directory or pass --force.")
    result = experiment()
    payload = json.dumps(result, indent=2, allow_nan=False)
    template = Path(__file__).with_name("report.html").read_text(encoding="utf-8")
    html = template.replace("/*__RESULTS__*/", payload.replace("<", "\\u003c"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(payload + "\n", encoding="utf-8")
    (args.out / "index.html").write_text(html, encoding="utf-8")
    print(f"Saved {args.out / 'index.html'} and results.json")


if __name__ == "__main__":
    main()
