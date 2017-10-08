from matplotlib import pyplot as plt
import csv
import argparse
import numpy as np


def read_log(logfile):
    timestep = []
    mean_reward = []
    best_mean_reward = []
    episodes = []
    exploration = []
    learning_rate = []

    with open(logfile, 'rt') as csvfile:
        logreader = csv.reader(csvfile, delimiter=' ')
        for row in logreader:
            if row[0] == 'Timestep':
                timestep.append(int(row[-1]))
            elif row[0] == 'mean':
                mean_reward.append(float(row[-1]))
            elif row[0] == 'best':
                best_mean_reward.append(float(row[-1]))
            elif row[0] == 'episodes':
                episodes.append(int(row[-1]))
            elif row[0] == 'exploration':
                exploration.append(float(row[-1]))

    timestep = np.array(timestep)
    mean_reward = np.array(mean_reward)
    best_mean_reward = np.array(best_mean_reward)
    episodes = np.array(episodes)
    exploration = np.array(exploration)
    learning_rate = np.array(learning_rate)

    return timestep, mean_reward, best_mean_reward, episodes, exploration, learning_rate


def main():
    parser = argparse.ArgumentParser(description='Plot training log')
    parser.add_argument('-l', '--logfile', type=str, help="Path to log file")
    parser.add_argument('-t', '--title', type=str, help="Plot title")
    parser.add_argument('--mean_reward', action='store_true', help="Add flag to plot mean reward")
    parser.add_argument('--best_mean_reward', action='store_true', help="Add flag to plot best mean reward")
    parser.add_argument('--episodes', action='store_true', help="Add flag to plot episodes")
    parser.add_argument('--exploration', action='store_true', help="Add flag to plot episodes")
    parser.add_argument('--learning_rate', action='store_true', help="Add flag to plot episodes")

    args = parser.parse_args()

    assert (args.mean_reward or args.best_mean_reward or args.episodes or args.exploration or args.learning_rate)

    timestep, mean_reward, best_mean_reward, episodes, exploration, learning_rate = read_log(args.logfile)

    plt.style.use('ggplot')
    plt.figure()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    if args.mean_reward:
        plt.plot(timestep, mean_reward, label='mean_reward')
    if args.best_mean_reward:
        plt.plot(timestep, best_mean_reward, label='best_mean_reward')
    if args.episodes:
        plt.plot(timestep, episodes, label='episodes')
    if args.exploration:
        plt.plot(timestep, exploration, label='exploration')
    if args.learning_rate:
        plt.plot(timestep, learning_rate, label='learning_rate')
    plt.legend()
    if args.title:
        plt.title(args.title)
    plt.xlabel('time steps')
    plt.show()


def plot_hw():
    logfiles = ["logBreakOut_ex", "logBreakOut", "logBreakOut_ex2"]
    timesteps = [read_log(logfile)[0] for logfile in logfiles]
    mean_rewards = [read_log(logfile)[1] for logfile in logfiles]
    best_mean_rewards = [read_log(logfile)[2] for logfile in logfiles]
    explorations = [read_log(logfile)[4] for logfile in logfiles]
    labels = ["original exploration schedule",
              "constant schedule (=0.05)",
              "aggresive schedule (start at 5, end at 0.005)"]

    plt.style.use('ggplot')
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(3, sharex=True)
    plt.subplots_adjust(hspace=0.3)
    for i in range(len(logfiles)):
        axarr[0].plot(timesteps[i], mean_rewards[i], label=labels[i])
    axarr[0].set_title('Breakout learning curve (mean reward) for different exploration schedules', fontsize=8)
    axarr[0].legend(fontsize=6)
    for i in range(len(logfiles)):
        axarr[1].plot(timesteps[i], best_mean_rewards[i], label=labels[i])
    axarr[1].set_title('Breakout learning curve (best mean reward) for different exploration schedules', fontsize=8)
    axarr[1].legend(fontsize=6)
    for i in range(len(logfiles)):
        axarr[2].plot(timesteps[i], explorations[i], label=labels[i])
    axarr[2].set_title('Breakout exploration schedules', fontsize=8)
    axarr[2].legend(fontsize=6)

    f.savefig("breakout.pdf")


if __name__ == '__main__':
    # main()
    plot_hw()