import sys
import numpy as np
import matplotlib.pyplot as plt
import parseREADME as parseRM

def phi(Um):
    GI = 2
    EI = -75
    rUm = 1/(1+np.exp(0.3*(-58-Um)))
    return rUm


def calc_r_teach(Um, Vis, TIMEBINS, RECORDING_STEP):
    Um = Um.split(", ")
    Um.pop()

    Um = np.array(list(map(lambda x: float(x), Um)))
    Um = Um.reshape((int(Um.size/(Vis)),Vis))

    teacher = phi(Um)
    teacher = teacher.T
    teacher = teacher[0:Vis,0:TIMEBINS//RECORDING_STEP]
    return teacher


def read_r_out(rU, Vis, Hid, hidden_recorded=False):
    rU = rU.split(", ")
    rU.pop()

    rU = np.array(list(map(lambda x: float(x), rU)))

    if hidden_recorded:
        rU = rU.reshape((rU.size//(Vis+Hid),Vis+Hid))
    else:
        rU = rU.reshape((rU.size//(Vis),Vis))

    rU = rU.T
    rU = rU[0:Vis,:]
    return rU


def mse(r_out, teach):
    err = np.square(r_out - teach).mean()
    return err, 0     # second return value is exact score for exact match


#def covariance(r_out, teach):
#    nrn_covs = 0
#    reference = 0
#    for nrn in range(r_out.shape[0]):
#        cov = np.cov(r_out[nrn, :], teach[nrn, :])
#        nrn_covs += cov[0, 1]
#        reference += cov[1, 1]
#    return nrn_covs, reference


#def cross_corr(r_out, teach):
#    nrn_covs = 0
#    reference = 0
#    for nrn in range(r_out.shape[0]):
#        corr = np.correlate(r_out[nrn, :], teach[nrn, :])
#        nrn_covs += corr
#        reference += np.correlate(teach[nrn, :], teach[nrn, :])
#    return nrn_covs, reference


def corr_coeff(r_out, teach):
    nrn_coeffs = 0
    reference = 0
    for nrn in range(r_out.shape[0]):
        coeff = np.corrcoef(r_out[nrn, :], teach[nrn, :])
        nrn_coeffs += coeff[0, 1]
        reference += coeff[1, 1]
    return nrn_coeffs, reference


#def corr_coeff_flat(r_out, teach):
#    coeff = np.corrcoef(r_out.flatten(), teach.flatten())
#    return coeff[0, 1], coeff[1, 1]


def plot_train_progress(r_teach, r_out, prefix, n_train_patterns, method, recalc, REC_ONE_NUDGED, TIMEBINS, RECORDING_STEP, eval_freq):
    eval_methods = {
        'mse': mse,
        'corr_coeff': corr_coeff,
    }
    try:
        if recalc:
            raise FileNotFoundError
        perf = np.load(prefix + 'train_progress_{}.npy'.format(method))
        reference = np.load(prefix + 'train_progress_{}_reference.npy'.format(method))
        print('Train progress data for {} already exists, loading old file'.format(method))
    except FileNotFoundError:
        print('Train progress data for {} does not exist, start calculating'.format(method))
        perf = []
        if REC_ONE_NUDGED:
            pattern_max = n_train_patterns * 2
            step = 2
        else:
            pattern_max = n_train_patterns
            step = 1
        for i in range(1, pattern_max, step):
            start = i * TIMEBINS // RECORDING_STEP
            end = (i + 1) * TIMEBINS // RECORDING_STEP
            r = r_out[:, start:end]
            result = eval_methods[method](r, r_teach)
            perf.append(result[0])
        reference = result[1]
        np.save(prefix + 'train_progress_{}.npy'.format(method), perf)
        np.save(prefix + 'train_progress_{}_reference.npy'.format(method), reference)

    plt.figure()
    plt.plot(range(0, n_train_patterns * eval_freq, eval_freq), perf, label=method)
    plt.axhline(reference, label='perfect match', ls='--')
    plt.legend()
    plt.xlabel('training cycle')
    plt.ylabel('performance')
    plt.savefig(prefix + 'train_progress_{}.png'.format(method))
    return


def plot_replay_decay(r_teach, r_out, prefix, n_train_pattern, n_replay, method, recalc, N_FINAL_NUDGE, TIMEBINS, RECORDING_STEP):
    eval_methods = {
        'mse': mse,
        'corr_coeff': corr_coeff,
    }
    try:
        if recalc:
            raise FileNotFoundError
        perf = np.load(prefix + 'replay_decay_{}.npy'.format(method))
        reference = np.load(prefix + 'replay_decay_{}_reference.npy'.format(method))
        print('Replay decay data already exists, loading old file')
    except FileNotFoundError:
        print('Replay data does not exist, start calculating')
        perf = []
        shifts = []
        last_shift = 0
        shift_tolerance = 100
        start_free_replay = n_train_pattern * 2 + N_FINAL_NUDGE + 1     # there seems to be one more nudged recording at the end of training
        max_shift = TIMEBINS // RECORDING_STEP                          # prevent from shifting fully to next pattern
        all_shifts = np.arange(0, max_shift, 1)
        for i in range(start_free_replay, start_free_replay + n_replay - N_FINAL_NUDGE - 1, 1):     # can't do last sample because shift not possible
            start = i * TIMEBINS // RECORDING_STEP
            end = (i + 1) * TIMEBINS // RECORDING_STEP
            shift_res = []
            for s in range(0, max_shift, 1):
                r = r_out[:, (start+s):(end+s)]
                result = eval_methods[method](r, r_teach)
                shift_res.append(result[0])
            if method == 'mse' or method == 'mse_norm':
                best_idx = np.argmin(shift_res)
            else:
                best_idx = np.argmax(shift_res)
            perf.append(shift_res[best_idx])
            shifts.append(best_idx)
        reference = result[1]
        np.save(prefix + 'replay_decay_{}.npy'.format(method), perf)
        np.save(prefix + 'replay_decay_{}_reference.npy'.format(method), reference)

    plt.figure()
    plt.plot(perf, label=method)
    plt.axhline(reference, label='perfect match', ls='--')
    plt.legend()
    plt.title('Replay decay after training')
    plt.xlabel('replay cycle')
    plt.ylabel('performance')
    plt.savefig(prefix + 'replay_decay_{}.png'.format(method))


if __name__ == '__main__':
    RECORDING_STEP = 2  # voltages/rates recorded very X timestep
    N_FINAL_NUDGE = 3   # how many patterns is vis clamped during final replay
    REC_ONE_NUDGED = True # was a nudged pattern recorded before every free episode during training
    HIDDEN_REC = False  # was the hidden layer recorded

    prefix = sys.argv[1]
    recalc_data = False  # if data was already calculated, just plot

    if len(sys.argv) > 2:
        if 'log_neptune' in sys.argv[2:]:
            log_neptune = True
        if 'recalc' in sys.argv[2:]:
            recalc_data = True
        if 'no_show' in sys.argv[2:]:
            no_show = True

    ruF = prefix + "ratesU.dat"
    umF = prefix + "Um.dat"
    fileRM = prefix + 'README'

    with open(fileRM, 'r') as f:
        dataVar = f.read()

    with open(ruF, 'r') as f:
        rU = f.read()

    with open(umF, 'r') as f:
        Um = f.read()

    CYCLES, TIMEBINS, Vis, Hid, RC, seed, HIDDEN_REC, p, q, eta_v, eta_h = parseRM.parseRM(dataVar)

    r_teach = calc_r_teach(Um, Vis, TIMEBINS, RECORDING_STEP)
    r_out = read_r_out(rU, Vis, Hid, HIDDEN_REC)
    print('n rec patterns', len(r_out[0]) // (TIMEBINS // RECORDING_STEP))

    # calc number of recorded eval cycles
    n_samples = r_out.shape[1]
    n_patterns = n_samples // TIMEBINS * RECORDING_STEP
    n_train_patterns = n_patterns - RC
    if REC_ONE_NUDGED:
        n_train_patterns //= 2
    eval_freq = CYCLES // n_train_patterns

    plot_train_progress(r_teach, r_out, prefix, n_train_patterns, 'corr_coeff', recalc_data, REC_ONE_NUDGED, TIMEBINS, RECORDING_STEP, eval_freq)
    plot_train_progress(r_teach, r_out, prefix, n_train_patterns, 'mse', recalc_data, REC_ONE_NUDGED, TIMEBINS, RECORDING_STEP, eval_freq)

    plot_replay_decay(r_teach, r_out, prefix, n_train_patterns, RC, 'mse', recalc_data, N_FINAL_NUDGE, TIMEBINS, RECORDING_STEP)
    plot_replay_decay(r_teach, r_out, prefix, n_train_patterns, RC, 'corr_coeff', recalc_data, N_FINAL_NUDGE, TIMEBINS, RECORDING_STEP)

    plt.show()
