import matplotlib.pyplot as plt

def plot(scores, mean_scores):
    plt.plot(scores, color='b')
    plt.plot(mean_scores, color='r')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.title('Snake Game Training')
    plt.grid(True)
    plt.show()
