import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure(1, figsize=(16, 5))
    plt.clf()
    plt.subplot(131)
    plt.plot(history['acc'])
    # plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(132)
    plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(133)
    plt.plot(history['reward'])
    plt.title('evaluation')
    plt.ylabel('evaluation')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')

    plt.tight_layout()
    plt.pause(0.1)
    pass

def plot_durations(rewards, means, save=False):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards)
    plt.plot(means)
    # plt.ylim(-400, 400)
    if save:
        plt.savefig("C:\\wspace\\data\\nn_tests\\{}.png".format("last_learn"))
    plt.pause(0.01)