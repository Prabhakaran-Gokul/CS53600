import tyro 
from cs536.assignment_5 import all_gather, broadcast
from matplotlib import pyplot as plt


def main():
    fig, axes = plt.subplots(2,2)
    size_bytes = [1 << s for s in range(10, 27, 2)]
    world_sizes = [2, 4, 8, 16, 32]

    #size_bytes = [1 << s for s in range(10, 16, 2)]
    #world_sizes = [2, 4]

    axes[0][0] = all_gather.msg_size_benchmark(axes[0][0], size_bytes)
    axes[0][1] = all_gather.rank_benchmark(axes[0][1], world_sizes)

    axes[1][0] = broadcast.msg_size_benchmark(axes[1][0], size_bytes)
    axes[1][1] = broadcast.rank_benchmark(axes[1][1], world_sizes)
    plt.tight_layout()
    plt.show()


'''
python -m cs536.assignment_5.run

'''

if __name__ == "__main__":
    tyro.cli(main)