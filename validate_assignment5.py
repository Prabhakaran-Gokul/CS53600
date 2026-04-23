import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cs536.assignment_5 import all_gather as ag
from src.cs536.assignment_5 import broadcast as bc


def validate_correctness() -> None:
    print("== Correctness checks ==")
    os.environ[ag.VALIDATE_ENV] = "1"
    os.environ[bc.VALIDATE_ENV] = "1"

    try:
        for world_size in (2, 4, 8):
            for fn in (ag.ring, ag.recursive_doubling, ag.swing):
                _ = ag.run_once(fn, world_size, 8)
                print(f"AllGather OK: fn={fn.__name__}, world_size={world_size}")

        for world_size in (2, 4, 8):
            for fn in (bc.binary_tree, bc.binomial_tree):
                _ = bc.run_once(fn, world_size, 8)
                print(f"Broadcast OK: fn={fn.__name__}, world_size={world_size}")
    finally:
        os.environ.pop(ag.VALIDATE_ENV, None)
        os.environ.pop(bc.VALIDATE_ENV, None)


def validate_smoke_benchmarks() -> None:
    print("== Smoke benchmark checks ==")

    fig, ax = plt.subplots()
    ag.msg_size_benchmark(ax, size_bytes=[16, 64, 256], world_size=8)
    plt.close(fig)
    print("AllGather msg-size sweep OK")

    fig, ax = plt.subplots()
    ag.rank_benchmark(ax, world_sizes=[2, 4, 8], msg_size=256)
    plt.close(fig)
    print("AllGather rank sweep OK")

    fig, ax = plt.subplots()
    bc.msg_size_benchmark(ax, size_bytes=[16, 64, 256], world_size=8)
    plt.close(fig)
    print("Broadcast msg-size sweep OK")

    fig, ax = plt.subplots()
    bc.rank_benchmark(ax, world_sizes=[2, 4, 8], msg_size=256)
    plt.close(fig)
    print("Broadcast rank sweep OK")


def validate_sequential_stability() -> None:
    print("== Sequential run_once stability checks ==")
    cases = [
        ("ag.ring", lambda: ag.run_once(ag.ring, 8, 8)),
        ("ag.recursive_doubling", lambda: ag.run_once(ag.recursive_doubling, 8, 8)),
        ("ag.swing", lambda: ag.run_once(ag.swing, 8, 8)),
        ("bc.binary_tree", lambda: bc.run_once(bc.binary_tree, 8, 8)),
        ("bc.binomial_tree", lambda: bc.run_once(bc.binomial_tree, 8, 8)),
    ]

    for round_idx in range(1, 4):
        for name, fn in cases:
            _ = fn()
            print(f"Sequential OK: round={round_idx}, fn={name}")


def main() -> None:
    validate_correctness()
    validate_smoke_benchmarks()
    validate_sequential_stability()
    print("All assignment 5 validations passed.")


if __name__ == "__main__":
    main()
