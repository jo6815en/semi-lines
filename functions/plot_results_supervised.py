import matplotlib.pyplot as plt
import re

# Define the path to your .log file
log_file_path = "./exp/finnwoods/supervised/supervised_only_finnwoods_from_scratch_2/all/20231025_105812.log"

# Lists to store the step and loss values
steps = []
losses = []
fscores = []
sAPs = []
epochs = []

# Define a regular expression pattern to search for step and loss values
pattern_loss = r"Iters: (\d+), Total loss: ([\d.]+)"
pattern_f = r"f_score:\s([0-9.]+),\s.*sAP10:\s([0-9.]+)"
# Read the .log file and extract step and loss values using regex
with open(log_file_path, "r") as file:
    for line in file:
        epoch = re.search(r"Epoch: (\d+)", line)
        f_match = re.search(pattern_f, line)
        match = re.search(pattern_loss, line)
        if epoch:
            found_epoch = int(epoch.group(1))
            epochs.append(found_epoch)
        if match:
            step = int(match.group(1))/3 + found_epoch * 10
            loss = float(match.group(2))
            steps.append(step)
            losses.append(loss)
        if f_match:
            fscore = float(f_match.group(1))
            sAP = float(f_match.group(2))
            fscores.append(fscore)
            sAPs.append(sAP)

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, label="Loss")
plt.title("Loss Plot")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Show the plot (or save it to a file if needed)
plt.show()

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, fscores, label="Fscore")
plt.title("Fscore Plot")
plt.xlabel("Epoch")
plt.ylabel("Fscore")
plt.legend()
plt.grid(True)

# Show the plot (or save it to a file if needed)
plt.show()

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, sAPs, label="sAP10")
plt.title("sAP10 Plot")
plt.xlabel("Epoch")
plt.ylabel("sAP10")
plt.legend()
plt.grid(True)

# Show the plot (or save it to a file if needed)
plt.show()