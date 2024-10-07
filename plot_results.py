import matplotlib.pyplot as plt
import re

# Define the path to your .log file
# log_file_path = "./exp/finnwoods/unimatch/unimatch_only_labeled/with_katam/20231102_085931.log"
log_file_path = "./exp/finnwoods/unimatch/unimatch_1_2_snoge_no_cutmix_smaller_lr/sam_segs_1_2/20240129_092501.log"



# Lists to store the step and loss values
steps = []
losses = []
losses_x = []
losses_s = []
fscores = []
sAPs = []
epochs = []

# Lists to store additional metrics
total_losses = []
c_values = []
d_values = []
l_values = []
junc_values = []
m_values = []
m_r_values = []

# Define a regular expression pattern to search for step and loss values
pattern_loss = r"Iters: (\d+), Total loss: ([\d.]+), Loss x: ([\d.]+), Loss s: ([\d.]+), Mask ratio: ([\d.]+), avg = ([\d.]+), c: ([\d.]+), d: ([\d.]+), l: ([\d.]+), junc:([\d.]+), m:([\d.]+), m_r:([\d.]+)"
pattern_f = r"f_score: ([\d.]+), recall: ([\d.]+), precision:([\d.]+), sAP10: ([\d.]+)"
pattern_epoch = r"Epoch: (\d+)"
# Read the .log file and extract step and loss values using regex
with open(log_file_path, "r") as file:
    for line in file:
        epoch_match = re.search(pattern_epoch, line)
        f_match = re.search(pattern_f, line)
        loss_match = re.search(pattern_loss, line)
        if epoch_match:
            found_epoch = int(epoch_match.group(1))
            epochs.append(found_epoch)
        if loss_match:
            step = int(loss_match.group(1))/225 + found_epoch * 5
            loss = float(loss_match.group(2))
            loss_x = float(loss_match.group(3))
            loss_s = float(loss_match.group(4))
            total_loss = float(loss_match.group(5))
            c_val = float(loss_match.group(6))
            d_val = float(loss_match.group(7))
            l_val = float(loss_match.group(8))
            junc_val = float(loss_match.group(9))
            m_val = float(loss_match.group(10))
            m_r_val = float(loss_match.group(11))

            steps.append(step)
            losses.append(loss)
            losses_x.append(loss_x)
            losses_s.append(loss_s)
            c_values.append(c_val)
            d_values.append(d_val)
            l_values.append(l_val)
            junc_values.append(junc_val)
            m_values.append(m_val)
            m_r_values.append(m_r_val)

        if f_match:
            print("In f_match")
            fscore = float(f_match.group(1))
            sAP = float(f_match.group(2))
            fscores.append(fscore)
            sAPs.append(sAP)

# Create a plot for total loss and components
plt.figure(figsize=(10, 6))
plt.plot(steps, c_values, label="c")
plt.plot(steps, d_values, label="d")
plt.plot(steps, l_values, label="l")
plt.plot(steps, junc_values, label="junc")
plt.plot(steps, m_values, label="m")
plt.plot(steps, m_r_values, label="m_r")

plt.title("Loss and Components Plot")
plt.xlabel("Step")
plt.ylabel("Values")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, label="Total Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(steps, losses_x, label="Loss x")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(steps, losses_s, label="Consistency loss")
plt.legend()
plt.grid(True)
plt.show()

# Create a plot for Fscore
plt.figure(figsize=(10, 6))
plt.plot(epochs, fscores, label="Fscore")
plt.title("Fscore Plot")
plt.xlabel("Epoch")
plt.ylabel("Fscore")
plt.legend()
plt.grid(True)
plt.show()

# Create a plot for sAP10
plt.figure(figsize=(10, 6))
plt.plot(epochs, sAPs, label="sAP10")
plt.title("sAP10 Plot")
plt.xlabel("Epoch")
plt.ylabel("sAP10")
plt.legend()
plt.grid(True)
plt.show()
