import numpy as np
import csv

# Read the Q-table from the .npy file
Q = np.load('q1.npy')

# Save the Q-table as a CSV file
with open('q1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(Q)