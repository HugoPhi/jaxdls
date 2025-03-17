import matplotlib.pyplot as plt


def plot_curve(acc, tacc, loss, tloss, epochs):
    fig, ax1 = plt.subplots()

    plt.rcParams['font.family'] = 'Noto Serif SC'
    plt.rcParams['font.sans-serif'] = ['Noto Serif SC']

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(range(epochs), acc, color=color, label='Train Accuracy', linestyle='-')
    ax1.plot(range(epochs), tacc, color=color, label='Test Accuracy', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(range(epochs), loss, color=color, label='Train Loss', linestyle='-')
    ax2.plot(range(epochs), tloss, color=color, label='Test Loss', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='lower right')

    plt.title('Training and Testing Accuracy and Loss over Epochs')
    fig.tight_layout()
    plt.show()

    print(f'final train, test acc : {acc[-1]}, {tacc[-1]}')
    print(f'final train, test loss: {loss[-1]}, {tloss[-1]}')
