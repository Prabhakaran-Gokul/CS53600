import tyro 
from cs536.assignment_2.throughput import run
from cs536.assignment_2.congestion_prediction import predict_next_cwnd
"""
Runs all of assignment 2
"""


'''
PYTHONPATH=src python -m cs536.assignment_2.run --file "src/cs536/assignment_2/results/ip_addresses.txt" --n 5 --duration 60 --interval 1 --verbose
'''

def main(file : str = "", n : int = 2, duration: int = 10, interval: float = 1.0, 
        verbose: bool = False):
    run(file, n, duration, interval, verbose, True, True)
    predict_next_cwnd()

    

if __name__ == "__main__":
    tyro.cli(main)
