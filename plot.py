#%%
import json
import matplotlib.pyplot as plt

def plot_loss_data(filename, label=None):
    # Load JSON data from file
    with open(filename, 'r') as file:
        data = json.load(file)

    # Extract iterations and loss values
    iterations = [entry[1] for entry in data['Loss']]
    loss_values = [entry[2] for entry in data['Loss']]

    # Create the plot
    plt.plot(iterations, loss_values, marker='o', label=label)
    plt.title('Loss vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
# %%
plot_loss_data("results/fed_ncf_50_15_512_0.01_50_1e-05_10.json")
plt.show()
# %%
plot_loss_data("results/fed_mf_100_15_512_0.1_50_1e-05_10.json")
plt.show()
# %%
plot_loss_data("results/fed_ncf_25_15_512_0.01_50_1e-05_10.json")
plot_loss_data("results/fed_ncf_50_15_512_0.01_50_1e-05_10.json")
plot_loss_data("results/fed_ncf_100_15_512_0.01_50_1e-05_10.json")
plot_loss_data("results/fed_ncf_500_15_512_0.01_50_1e-05_10.json")
plt.legend(["25", "50", "100", "500"], title="Number of Workers")
plt.show()
# %%
