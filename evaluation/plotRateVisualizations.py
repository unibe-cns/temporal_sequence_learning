import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import parseREADME as parseRM


# define x-label formatter
def formatXLabelTC(x, pos=None):
    return str(x/10*2)


def phi(Um):
    GI = 2
    EI = -75
    rUm = 1/(1+np.exp(0.3*(-58-Um)))
    return rUm


def plot_full_training(rU):
    fig = plt.figure("Raster Rate Plot Full Run")
    img = plt.imshow(rU.T, aspect='auto', interpolation='none')
    print('Shape of recorded rates, full training', (rU.T).shape)
    img.set_cmap('GnBu')
    plt.title("Raster Rate Plot Full Run")
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(formatXLabelTC))
    plt.gca().invert_yaxis()
    plt.axvline(x=(((rU.T).shape[1])-(RC*TIMEBINS//RECORDING_STEP)), linewidth=2, color='#b134eb')
    plt.xlabel("Time [ms]")
    plt.ylabel("Neuron")
    fig.tight_layout()
    plt.savefig(prefix + 'rates_full_training.png')


def plot_final_replay(rU, all_zooms=True):
    fig = plt.figure("Raster Rate Plot Final Replay")
    img = plt.imshow(rU.T, aspect='auto', interpolation='none')
    plt.title("Raster Rate Plot Final Replay")
    img.set_cmap('GnBu')
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(formatXLabelTC))
    plt.gca().invert_yaxis()
    plt.axvline(x=(((rU.T).shape[1])-(RC*TIMEBINS//RECORDING_STEP)), linewidth=2, color='#b134eb')
    final_replay_start = (rU.T).shape[1] - RC*TIMEBINS//RECORDING_STEP
    final_replay_end = (rU.T).shape[1]
    end_nudging = final_replay_start + N_FINAL_NUDGE * TIMEBINS//RECORDING_STEP
    plt.axvline(x=end_nudging, linewidth=2, color='red')
    plt.xlim(final_replay_start, final_replay_end)
    plt.xlabel("Time [ms]")
    plt.ylabel("Neuron")
    fig.tight_layout()
    plt.savefig(prefix + 'rates_final_replay.png')

    if all_zooms:
        fig = plt.figure("Raster Rate Plot Final Replay (Zoom Start)")
        img = plt.imshow(rU.T, aspect='auto', interpolation='none')
        img.set_cmap('GnBu')
        plt.title("Raster Rate Plot Final Replay (Zoom Start)")
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(formatXLabelTC))
        plt.gca().invert_yaxis()
        plt.axvline(x=(((rU.T).shape[1])-(RC*TIMEBINS//RECORDING_STEP)), linewidth=2, color='#b134eb')
        final_replay_start = (rU.T).shape[1] - RC*TIMEBINS//RECORDING_STEP
        final_replay_end = (rU.T).shape[1] - (RC - 8)*TIMEBINS//RECORDING_STEP
        end_nudging = final_replay_start + N_FINAL_NUDGE * TIMEBINS//RECORDING_STEP
        plt.axvline(x=end_nudging, linewidth=2, color='red')
        plt.xlim(final_replay_start, final_replay_end)
        plt.xlabel("Time [ms]",fontsize=14)
        plt.ylabel("Neuron",fontsize=14)
        fig.tight_layout()
        plt.savefig(prefix + 'rates_final_replay_zoom_start.png')

        fig = plt.figure("Raster Rate Plot Final Replay (Zoom End)")
        img = plt.imshow(rU.T, aspect='auto', interpolation='none')
        plt.title("Raster Rate Plot Final Replay (Zoom End)")
        img.set_cmap('GnBu')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(formatXLabelTC))
        plt.gca().invert_yaxis()
        final_replay_start = (rU.T).shape[1] - 5*TIMEBINS//RECORDING_STEP
        final_replay_end = (rU.T).shape[1]
        end_nudging = final_replay_start + N_FINAL_NUDGE * TIMEBINS//RECORDING_STEP
        plt.xlim(final_replay_start, final_replay_end)
        plt.xlabel("Time [ms]")
        plt.ylabel("Neuron")
        fig.tight_layout()
        plt.savefig(prefix + 'rates_final_replay_zoom_end.png')
    return


if __name__ == '__main__':
    nudgeOFF = 10       # ? currently not used?
    RECORDING_STEP = 2  # voltages/rates recorded very X timestep
    N_FINAL_NUDGE = 3   # how many patterns is vis clamped during final replay

    prefix = sys.argv[1]
    ruF = prefix + "ratesU.dat"
    fileRM = prefix + 'README'

    with open(fileRM, 'r') as f:
        dataVar = f.read()

    with open(ruF, 'r') as f:
        rU = f.read()

    CYCLES, TIMEBINS, Vis, Hid, RC, seed, rec_hid, p, q, etav, etah = parseRM.parseRM(dataVar)

    rU = rU.split(", ")
    rU.pop()
    rU = np.array(list(map(lambda x: float(x), rU)))

    if rec_hid:
        rU = rU.reshape((rU.size//(Vis+Hid),Vis+Hid))
    else:
        rU = rU.reshape((rU.size//(Vis),Vis))

    plot_full_training(rU)
    plot_final_replay(rU, all_zooms=True)

    plt.show()

