import numpy as np
from tct.tct_elastic import tct_elastic_generate_u_interface


def generate_training_set():
    frequency_range = range(500, 2000, int(1500/20))

    training_in = []
    training_out = []
    for frequency in frequency_range:
        print("Running Simulation for frequency: ", frequency)
        u, t = tct_elastic_generate_u_interface(frequency)
    
        training_in.append(t)
        training_out.append(u)

    with open("./training_in.npy", "wb") as f:
        np.save(f, np.array(training_in))

    with open("./training_out.npy", "wb") as f:
        np.save(f, np.array(training_out))


if __name__ == "__main__":
    generate_training_set()
