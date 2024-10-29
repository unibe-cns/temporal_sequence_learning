import sys
import numpy as np
import matplotlib.pyplot as plt
import parseREADME as parseRM


def plot_som_dend_weights(pre_train=False):
    if pre_train:
        w = W_pre.split(", ")
    else:
        w = W.split(", ")
    w.pop()
    w = np.array(list(map(lambda x: float(x), w)))
    W2 = w.reshape((w.size//(Vis+Hid),Vis+Hid))
    #print("som-dend weight matrix", W2.shape, "dim-0: postsyn nrn, dim-1: presyn nrn")
    #print("Starts with visible neurons")

    if pre_train:
        fig_pre = plt.figure(figsize=(7,7))
        plt.imshow(W2[-(Vis+Hid):,:], interpolation='none',\
                cmap="bwr", vmin=-W2.max(),vmax=W2.max())
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=22)
        plt.title("Weight Matrix (Pre training)", fontsize=28)
        name = 'weight_matrix_before_learning.png'
        np.save(prefix + 'numpy_pre_train_soma_dend_weights.npy', W2)
    else:
        fig = plt.figure(figsize=(7,7))
        plt.imshow(W2[-(Vis+Hid):,:], interpolation='none',\
                cmap="bwr", vmin=-W2.max(),vmax=W2.max())
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=22)
        plt.title("Weight Matrix (Post training)", fontsize=28)
        name = 'weight_matrix_after_learning.png'
        np.save(prefix + 'numpy_post_train_soma_dend_weights.npy', W2)

    plt.xlabel('Presynaptic Neuron',fontsize=25)
    plt.ylabel('Postsynaptic Neuron',fontsize=25)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(-0.5,Vis+Hid-0.5)
    plt.xlim(-0.5,Vis+Hid-0.5)
    plt.axhline(Vis - 0.5)
    plt.axvline(Vis - 0.5)

    if pre_train:
        fig_pre.tight_layout()
        fig_pre.savefig(prefix + name)
    else:
        fig.tight_layout()
        fig.savefig(prefix + name)

    text = 'Post training'
    if pre_train:
        text = 'Pre training'
    print('Weight distribution ({}):'.format(text))
    print('Vis->Vis mu: {0} std: {1}'.format(np.mean(W2[0:Vis, 0:Vis]), np.std(W2[0:Vis, 0:Vis])))
    print('Vis->Hid mu: {0} std: {1}'.format(np.mean(W2[Vis:, 0:Vis]), np.std(W2[Vis:, 0:Vis])))
    print('Hid->Hid mu: {0} std: {1}'.format(np.mean(W2[Vis:, Vis:]), np.std(W2[Vis:, Vis:])))
    print('Hid->Vis mu: {0} std: {1}'.format(np.mean(W2[0:Vis, Vis:]), np.std(W2[0:Vis, Vis:])))
    fig, ax = plt.subplots(1, 1)
    plt.title('soma->dend weight distributions ({})'.format(text))
    counts, bins = np.histogram(W2[Vis:, Vis:])
    ax.hist(bins[:-1], bins, weights=counts, label='Hid->Hid: mu {0:.4f} sig {1:.4f}'.format(np.mean(W2[Vis:, Vis:]), np.std(W2[Vis:, Vis:])))
    counts, bins = np.histogram(W2[Vis:, 0:Vis])
    ax.hist(bins[:-1], bins, weights=counts, label='Vis->Hid: mu {0:.4f} sig {1:.4f}'.format(np.mean(W2[Vis:, 0:Vis]), np.std(W2[Vis:, 0:Vis])))
    plt.legend()
    if pre_train:
        name = 'soma_dend_weight_distr_pre_training.png'
    else:
        name = 'soma_dend_weight_distr_after_learning.png'
    plt.savefig(prefix + name)
    return


def plot_nudge_matrix(nNudgeOut, nNudgeIn, nudgeConns):
    nNudgeOut = nNudgeOut.split(", ")
    nNudgeOut.pop()
    nNudgeOut = np.array(list(map(lambda x: int(x), nNudgeOut)))
    nNudgeIn = nNudgeIn.split(", ")
    nNudgeIn.pop()
    nNudgeIn = np.array(list(map(lambda x: int(x), nNudgeIn)))
    nudges = nudgeConns.split(";")
    nudges.pop()
    nudges = [[int(elem.split(",")[0]), int(elem.split(",")[1])] for elem in nudges]
    matrix = np.zeros((Vis + Hid, Vis + Hid))
    for teach_idx, stud_idx in nudges:
        if matrix[stud_idx, teach_idx] == 1:
            print('Connection {0} -> {1} already formed!'.format(teach_idx, stud_idx))
        matrix[stud_idx, teach_idx] = 1
    assert(sum(nNudgeOut) == sum(nNudgeIn)), "sum(nNudgeOut)={0} sum(nNudgeIn)={1}".format(sum(nNudgeOut), sum(nNudgeIn))
    if not (matrix.sum() == sum(nNudgeIn)):
        sys.stderr.write("WARNING: matrix.sum()={0} sum(nNudgeIn)={1}\n".format(matrix.sum(), sum(nNudgeIn)))
        sys.stderr.write("nudgeConns: {0}\n".format(nudgeConns))

    print('nNudgeOut', nNudgeOut, 'num Elements: ', len(nNudgeOut))
    print('nNudgeIn', nNudgeIn, 'num Elements: ', len(nNudgeIn))
    print('Number of nudging connections formed: {}'.format(sum(nNudgeOut)))
    print('nudges', nudges)
    fig0 = plt.figure("Nudge Matrix", figsize=(7,7))
    plt.imshow(matrix[-(Vis+Hid):,:], interpolation='none', cmap="Blues")
    plt.title("Som-Som nudging", fontsize=28)
    np.save(prefix + 'nudging_matrix.npy', matrix)
    plt.xlabel('Presynaptic Neuron',fontsize=25)
    plt.ylabel('Postsynaptic Neuron',fontsize=25)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(-0.5,Vis+Hid-0.5)
    plt.xlim(-0.5,Vis+Hid-0.5)
    plt.axhline(Vis - 0.5)
    plt.axvline(Vis - 0.5)
    fig0.tight_layout()
    plt.savefig(prefix + 'nudging_matrix.png')


if __name__ == '__main__':
    prefix = sys.argv[1]
    wF = prefix + "weights.dat"
    wF_pre = prefix + "pre_train_weights.dat"
    nOutNudgeF = prefix + "nOut.dat"
    nInNudgeF = prefix + "nIn.dat"
    NudgeConnsF = prefix + "nudge.dat"
    fileRM = prefix + 'README'

    with open(fileRM, 'r') as f:
        dataVar = f.read()

    with open(wF, 'r') as f:
        W = f.read()

    with open(wF_pre, 'r') as f:
        W_pre = f.read()

    with open(nOutNudgeF, 'r') as f:
        nOutNudge = f.read()

    with open(nInNudgeF, 'r') as f:
        nInNudge = f.read()

    with open(NudgeConnsF, 'r') as f:
        nudgeConns = f.read()

    CYCLES, TIMEBINS, Vis, Hid, RC, seed, rec_hid, p, q, etav, etah = parseRM.parseRM(dataVar)

    plot_som_dend_weights(pre_train=True)
    plot_som_dend_weights()
    plot_nudge_matrix(nOutNudge, nInNudge, nudgeConns)

    plt.show()
