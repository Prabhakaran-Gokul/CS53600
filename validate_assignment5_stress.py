import os
import time

from src.cs536.assignment_5 import all_gather as ag
from src.cs536.assignment_5 import broadcast as bc


def run_stress(rounds: int = 4, msg_size: int = 16) -> None:
    all_gather_cases = [
        ("ag.ring", ag.ring, [2, 4, 8]),
        ("ag.recursive_doubling", ag.recursive_doubling, [2, 4, 8]),
        ("ag.swing", ag.swing, [2, 4, 8]),
    ]
    broadcast_cases = [
        ("bc.binary_tree", bc.binary_tree, [2, 4, 8]),
        ("bc.binomial_tree", bc.binomial_tree, [2, 4, 8]),
    ]

    os.environ[ag.VALIDATE_ENV] = "1"
    os.environ[bc.VALIDATE_ENV] = "1"

    try:
        for round_idx in range(1, rounds + 1):
            print(f"round {round_idx}/{rounds}")

            for name, fn, world_sizes in all_gather_cases:
                for world_size in world_sizes:
                    t0 = time.perf_counter()
                    elapsed = ag.run_once(fn, world_size, msg_size)
                    print(
                        f"  {name} ws={world_size} comm_time={elapsed:.6f}s "
                        f"wall={time.perf_counter() - t0:.3f}s"
                    )

            for name, fn, world_sizes in broadcast_cases:
                for world_size in world_sizes:
                    t0 = time.perf_counter()
                    elapsed = bc.run_once(fn, world_size, msg_size)
                    print(
                        f"  {name} ws={world_size} comm_time={elapsed:.6f}s "
                        f"wall={time.perf_counter() - t0:.3f}s"
                    )

        print("stress validation passed")
    finally:
        os.environ.pop(ag.VALIDATE_ENV, None)
        os.environ.pop(bc.VALIDATE_ENV, None)


def main() -> None:
    run_stress()


if __name__ == "__main__":
    main()
