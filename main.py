# Import required python libraries #
import time

import di_io

start_time_groups_main = time.time()  # Start the time.

# Ask the user to define the task #
di_io.read()

# Excecute python scripts#
exec(open("mass_size.py").read())

print("--- Finished main.py in %.5s seconds ---" % (time.time() - start_time_groups_main))  # Print groups time.