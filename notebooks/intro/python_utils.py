import scipy.io
import numpy as np
#import mat73
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import random
import matplotlib.colors as mcolors
import scipy.io
from mpl_toolkits import mplot3d
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import matplotlib as mpl
import matplotlib.colors as colors
import scipy.stats as stats
#import pandas as pd
import scipy
#from statannot import add_stat_annotation
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os
import datetime

from fooof import FOOOF
from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import colors

import networkx as nx
import numpy as np

def extract_topological_metrics(projs, n_exc, n_inh):
    """
    Estrae un grafo diretto dai projections PyNN e calcola metriche topologiche chiave.

    Args:
        projs (dict): dizionario con chiavi tuple (pre_pop, post_pop, tipo) e projection PyNN.
        n_exc (int): numero di neuroni eccitatori.
        n_inh (int): numero di neuroni inibitori.

    Returns:
        dict: metriche topologiche principali.
    """
    import networkx as nx
    import numpy as np

    G = nx.DiGraph()
    G.add_nodes_from(range(n_exc + n_inh))
    offset = {'exc': 0, 'inh': n_exc}

    for key in projs:
        pre_pop, post_pop, syn_type = key
        pre_offset = offset['exc'] if pre_pop == 'cx' and syn_type == 'exc' else offset['inh']
        post_offset = offset['exc'] if post_pop == 'cx' and syn_type == 'exc' else offset['inh']
        
        try:
            conn_list = projs[key].get(['source', 'target', 'weight'], format='list')
        except Exception as e:
            print(f"Errore su {key}: {e}")
            continue

        for src, tgt, weight in conn_list:
            G.add_edge(src + pre_offset, tgt + post_offset, weight=weight)

    metrics = {
        'density': nx.density(G),
        'avg_degree': np.mean([deg for _, deg in G.degree()]),
        'avg_clustering': nx.average_clustering(G.to_undirected()),
        'avg_shortest_path': nx.average_shortest_path_length(G) if nx.is_strongly_connected(G) else np.nan,
        'avg_in_degree': np.mean([deg for _, deg in G.in_degree()]),
        'avg_out_degree': np.mean([deg for _, deg in G.out_degree()])
    }

    return metrics


def plot_population_with_fooof(results, pop_label='exc', fileName='plot', saveDir='figures', fs=1000, nperseg=256, rate=4):
    features = ['v', 'gsyn_exc', 'gsyn_inh']
    labels = ['membrane potential [mV]', 'gsyn_exc [µS]', 'gsyn_inh [µS]']
    time = np.arange(results[pop_label, 'v'].shape[0])  # tempo in ms

    fig, ax = plt.subplots(4, 4, figsize=(18, 14), sharex=False)
    ax = ax.flatten()
    p = {}

    for idx, (feat, label) in enumerate(zip(features, labels)):
        data = np.asarray(results[pop_label, feat])

        # Heatmap
        norm = colors.Normalize(vmin=0, vmax=np.percentile(data, 99)) if 'gsyn' in feat else colors.PowerNorm(gamma=0.75)
        p[idx] = ax[idx].imshow(data.T, norm=norm, aspect='auto', origin='lower', cmap='viridis')
        fig.colorbar(p[idx], ax=ax[idx], shrink=0.6)
        ax[idx].set_title(label, fontsize=13)
        ax[idx].set_ylabel('neurons', fontsize=12)

        # Media ± std
        mean_trace = data.mean(axis=1)
        std_trace = data.std(axis=1)
        ax[idx + 4].plot(time, mean_trace, color='k', lw=2)
        ax[idx + 4].fill_between(time, mean_trace - std_trace, mean_trace + std_trace, color='gray', alpha=0.4)
        ax[idx + 4].set_title('Average ' + label, fontsize=13)
        ax[idx + 4].set_xlabel('time [ms]')
        ax[idx + 4].set_ylabel('mean ± std')

    # g_exc / (g_exc + g_inh)
    gexc = np.asarray(results[pop_label, 'gsyn_exc']).mean(axis=1)
    ginh = np.asarray(results[pop_label, 'gsyn_inh']).mean(axis=1)
    balance = gexc / (gexc + ginh + 1e-10)
    mean_balance = np.mean(balance)

    ax[3].plot(time, balance, color='purple', label='Balance')
    ax[3].axhline(mean_balance, color='black', linestyle='--', linewidth=2, label=f'Mean = {mean_balance:.3f}')
    ax[3].set_title('Synaptic Balance')
    ax[3].set_xlabel('time [ms]')
    ax[3].set_ylabel('g_exc / (g_exc + g_inh)')
    ax[3].set_ylim([0, 1])
    ax[3].legend(loc='lower right', fontsize=10)


    # Raster
    spikes = results.get((pop_label, 'spikes'), [])
    if spikes:
        # Calcolo del rate medio
        total_spikes = sum(len(s) for s in spikes)
        n_neurons = len(spikes)
        duration_s = time[-1] / 1000.0  # ms → s
        mean_rate = total_spikes / (n_neurons * duration_s)

        for neuron_id, spike_times in enumerate(spikes):
            ax[7].vlines(spike_times, neuron_id - 0.4, neuron_id + 0.4, color='black', linewidth=0.5)

        rate_str = f"{RATE} Hz" if 'RATE' in locals() else "unknown"
        ax[7].set_title(f'Raster Plot – Mean Rate = {mean_rate:.2f} Hz\nInput Poisson at {rate}')
    else:
        ax[7].text(0.5, 0.5, 'No spikes', ha='center', va='center')
        ax[7].set_title('Raster Plot')

    ax[7].set_xlabel('time [ms]')
    ax[7].set_ylabel('neuron ID')



    # --- PSD e FOOOF ---
    sig = np.asarray(results[pop_label, 'v']).T
    f, psd = welch(sig, fs=fs, axis=0, nperseg=nperseg)
    psd_mean = np.mean(psd, axis=1)
    psd_sem = np.std(psd, axis=1) / np.sqrt(psd.shape[1])

    fm = FOOOF(verbose=False)
    fm.fit(f, psd_mean)
    offset, exponent = fm.get_params('aperiodic_params')

    # PSD log-log
    ax[8].plot(f, psd_mean, c='black')
    ax[8].fill_between(f, psd_mean - psd_sem, psd_mean + psd_sem, color='gray', alpha=0.3)
    ax[8].set_xscale('log')
    ax[8].set_yscale('log')
    ax[8].set_title(f'PSD log-log\nAperiodic: offset={offset:.2f}, exponent={exponent:.2f}')
    ax[8].set_xlabel('Freq (Hz)')
    ax[8].set_ylabel('PSD')

    # PSD linear-linear
    ax[9].plot(f, psd_mean, c='black')
    ax[9].fill_between(f, psd_mean - psd_sem, psd_mean + psd_sem, color='gray', alpha=0.3)
    ax[9].set_title(f'PSD linear\nAperiodic: offset={offset:.2f}, exponent={exponent:.2f}')
    ax[9].set_xlabel('Freq (Hz)')
    ax[9].set_ylabel('PSD')

    # Nascondi pannelli inutilizzati
    for i in range(10, len(ax)):
        ax[i].axis('off')

    fig.suptitle(f'{pop_label.upper()} population summary', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.join(saveDir, fileName), exist_ok=True)
    fig.savefig(os.path.join(saveDir, fileName, f'{pop_label}_summary.png'), dpi=300)
    print(f"Figura salvata in: {os.path.join(saveDir, fileName, f'{pop_label}_summary.png')}")
    plt.show()


def plot_population(results, pop_label='exc', fileName='plot', saveDir='figures'):
    features = ['v', 'gsyn_exc', 'gsyn_inh']
    labels = ['membrane potential [mV]', 'gsyn_exc [µS]', 'gsyn_inh [µS]']
    time = np.arange(results[pop_label, 'v'].shape[0])  # tempo in ms

    fig, ax = plt.subplots(4, 3, figsize=(16, 13), sharex=False)
    ax = ax.flatten()
    p = {}

    for idx, (feat, label) in enumerate(zip(features, labels)):
        data = np.asarray(results[pop_label, feat])

        # Normalizzazione: gsyn con scala lineare, v con PowerNorm
        if 'gsyn' in feat:
            vmax = np.percentile(data, 99)
            norm = colors.Normalize(vmin=0, vmax=vmax)
        else:
            norm = colors.PowerNorm(gamma=0.75)

        # Heatmap
        p[idx] = ax[idx].imshow(data.T, norm=norm, aspect='auto', origin='lower', cmap='viridis')
        fig.colorbar(p[idx], ax=ax[idx], shrink=0.6)
        ax[idx].set_title(label, fontsize=13)
        ax[idx].set_ylabel('neurons', fontsize=12)

        # Media ± std
        mean_trace = data.mean(axis=1)
        std_trace = data.std(axis=1)

        ax[idx + 3].plot(time, mean_trace, color='k', lw=2)
        ax[idx + 3].fill_between(time, mean_trace - std_trace, mean_trace + std_trace, color='gray', alpha=0.4)
        ax[idx + 3].set_title('Average ' + label, fontsize=13)
        ax[idx + 3].set_xlabel('time [ms]', fontsize=12)
        ax[idx + 3].set_ylabel('mean ± std', fontsize=12)
        #if 'gsyn' in label: 
            #ax[idx + 3].set_ylim([0, 0.25])


   # Pannello bilancio sinaptico: gexc / (gexc + ginh)
    gexc = np.asarray(results[pop_label, 'gsyn_exc']).mean(axis=1)
    ginh = np.asarray(results[pop_label, 'gsyn_inh']).mean(axis=1)
    balance = gexc / (gexc + ginh + 1e-10)  # per evitare divisione per 0
    
    # Pannello 6: gsyn globali (media tra exc e inh)
    gexc_exc = np.asarray(results['exc', 'gsyn_exc']).mean(axis=1)
    ginh_exc = np.asarray(results['exc', 'gsyn_inh']).mean(axis=1)
    gexc_inh = np.asarray(results['inh', 'gsyn_exc']).mean(axis=1)
    ginh_inh = np.asarray(results['inh', 'gsyn_inh']).mean(axis=1)

    gexc_total = (gexc_exc + gexc_inh) / 2
    ginh_total = (ginh_exc + ginh_inh) / 2

    ax[6].plot(time, gexc_total, label='mean g_exc', color='blue')
    ax[6].plot(time, ginh_total, label='mean g_inh', color='red')
    ax[6].set_title('Global gsyn (exc + inh average)', fontsize=13)
    ax[6].set_xlabel('time [ms]')
    ax[6].set_ylabel('mean [µS]')
    ax[6].legend(fontsize=10)


    # Pannello 7: bilancio complessivo dalle medie globali
    balance_global = gexc_total / (gexc_total + ginh_total + 1e-10)
    mean_balance = np.mean(balance_global)
    ax[7].plot(time, balance_global, color='darkgreen', label='Balance over time')
    ax[7].axhline(mean_balance, color='black', linewidth=2.5, linestyle='--',
                  label=f'Mean = {mean_balance:.3f}')

    ax[7].set_title('Global Synaptic Balance', fontsize=13)
    ax[7].set_xlabel('Time [ms]')
    ax[7].set_ylabel('g_exc / (g_exc + g_inh)', fontsize=12)
    ax[7].set_ylim([0, 1])
    ax[7].legend(fontsize=10, loc='lower right')

    ax[8].plot(time, balance, color='purple', lw=2)
    ax[8].set_title('Synaptic Balance', fontsize=13)
    ax[8].set_xlabel('time [ms]', fontsize=12)
    ax[8].set_ylabel('g_exc / (g_exc + g_inh)', fontsize=12)
    ax[8].set_ylim([0, 1])

    # Pannello 9: raster plot (spike times per neurone)
    if (pop_label, 'spikes') in results:
        spikes = results[pop_label, 'spikes']  # formato: list of list
        for neuron_id, spike_times in enumerate(spikes):
            ax[9].vlines(spike_times, neuron_id - 0.4, neuron_id + 0.4, color='black', linewidth=0.5)
        ax[9].set_title('Raster Plot', fontsize=13)
        ax[9].set_xlabel('time [ms]', fontsize=12)
        ax[9].set_ylabel('neuron ID', fontsize=12)
        ax[9].set_xlim([0, time[-1]])
    else:
        ax[9].text(0.5, 0.5, 'No spike data found', ha='center', va='center', transform=ax[9].transAxes)
        ax[9].set_title('Raster Plot')

        for idx, (label, color) in enumerate(zip(['inh', 'exc'], ['b', 'r'])):
            feat = 'v'
            sig = np.asarray(results[label, feat]).T
            f, psd = welch(sig, fs=fs, axis=0, nperseg=nperseg)
            psd_mean = np.mean(psd, axis=1)
            psd_sem = np.std(psd, axis=1) / np.sqrt(psd.shape[1])
            axs[11 + idx].plot(f, psd_mean, label=f'{label.upper()} PSD', c=color)
            axs[11 + idx].fill_between(f, psd_mean - psd_sem, psd_mean + psd_sem, color=color, alpha=0.3, label=f'{label.upper()} ±1 SEM')
            axs[11 + idx].set_xscale('log')
            axs[11 + idx].set_yscale('log')
            axs[11 + idx].legend()
            axs[11 + idx].set_title(f'{label.upper()} Welch PSD')
            axs[11 + idx].set_xlabel('Freq (Hz)')
            axs[11 + idx].set_ylabel('PSD')
        
        
        
    fig.suptitle(f'{pop_label.upper()} population', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Nascondi pannelli inutilizzati (ax[10], ax[11])
    for i in [10, 11]:
        ax[i].set_visible(False)
        
    # === Salvataggio automatico ===
    output_dir = os.path.join(saveDir, f"{fileName}")
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{pop_label}_summary.png")
    fig.savefig(fig_path, dpi=300)
    print(f"Figura salvata in: {fig_path}")
    plt.show()



def extractSpike(spikeData):
    VAR=[]
    if len(spikeData)>=2:
        for k in range(len(spikeData)):
            VAR.append(np.array(list(spikeData[k])).T,)
            
    else: VAR.append(np.array(list(spikeData)).T,)
    return VAR


def recover_results(outputs, do_print=True):
    results = {}
    for key in outputs.keys(): # to extract the name of the layer, e.g., Exc, Inh, Thalamus, etc  
        if do_print: print(key)
        # to get voltage and conductances
        for analogsignal in outputs[key].segments[0].analogsignals:
            print(analogsignal.name)
            results[key, analogsignal.name] = analogsignal

        # to get spikes
        VAR=outputs[key].segments[0].spiketrains
        results[key, 'spikes']=[]
        for k in range(len(VAR)):
            results[key, 'spikes'].append(np.array(list(VAR[k])).T,)
    return results