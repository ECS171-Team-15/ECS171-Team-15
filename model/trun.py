import time
import subprocess
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: trun.py height width")
        print("   Test runner runs only one combination (dropout & kernel_regularizer) each time to avoid GPU memory issue.")
        exit(1)

    hi = int(sys.argv[1])
    wid = int(sys.argv[2])

    start_time = time.time()
    print("Current working directory:", os.getcwd())

    param_grid = {
        'dropout': [
            True, False
        ],
        'kernel_regularizer': [
            'l1_l2', None
        ]
    }

    runParm = {
        'ranNum': [
            6, 13, 45, 77
        ]
    }
    SP = " "
    for ran in runParm['ranNum']:
        for drop in param_grid['dropout']:
            for regu in param_grid['kernel_regularizer']:
                # cmd = r"C:\Users\delli7new2\Documents\GitHub\ECS171-Team-15\venv\Scripts\python.exe cnn_model.py"
                cmd = r"C:\Users\delli7new2\Documents\GitHub\ECS171-Team-15\venv\Scripts\python.exe cnn_MOD.py "
                cmd0 = f'{cmd} {hi} {wid} {ran} {drop} {regu}'
                print(" Running: " + cmd0)
                returned_value = os.system(cmd0)  # returns the exit code in unix
                print('returned value:', returned_value)

    print(f"\nProgram time: {time.time() - start_time} seconds")