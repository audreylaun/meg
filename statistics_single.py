
import argparse

import numpy as np
from numpy.random import randn
from scipy import stats as stats
import scipy
import matplotlib.pyplot as plt
from mne.viz import plot_compare_evokeds
import pandas as pd



import mne
from mne.datasets import sample
from mne.epochs import equalize_epoch_counts
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from mpl_toolkits.axes_grid1 import ImageGrid, inset_locator, make_axes_locatable
from mne.channels import find_ch_adjacency
from mne.stats import combine_adjacency, spatio_temporal_cluster_test



def statistics_single_sensor(sub):
    directory = '/Users/admin/Box Sync/Starling/Experiment1/MEG_data/' + sub + '/'
    tmin = 0.3
    tmax = 0.6

    #Load all epoched data
    prod_fname = directory + sub + '_prod_TEST-epo.fif'
    epochs_prod = mne.read_epochs(prod_fname, preload =True)
    comp_fname = directory + sub + '_comp_TEST-epo.fif'
    epochs_comp = mne.read_epochs(comp_fname, preload = True)

    #load and apply projectors
    proj_fname = directory + sub + '-proj.fif'
    proj = mne.read_proj(proj_fname)
    epochs_prod.add_proj(proj)
    epochs_comp.add_proj(proj)
    epochs_prod.apply_proj()
    epochs_comp.apply_proj()

    epochs_prod.interpolate_bads()
    epochs_comp.interpolate_bads()

    ident_prod = epochs_prod['production identical']
    unrel_prod = epochs_prod['production unrelated']
    ident_comp = epochs_comp['comprehension identical']
    unrel_comp = epochs_comp['comprehension unrelated']

    mne.epochs.equalize_epoch_counts([ident_prod, unrel_prod, ident_comp, unrel_comp], method = 'random')

    ident_prod = ident_prod.filter(l_freq=0, h_freq=40)
    unrel_prod = unrel_prod.filter(l_freq=0, h_freq=40)
    ident_comp = ident_comp.filter(l_freq=0, h_freq=40)
    unrel_comp = unrel_comp.filter(l_freq=0, h_freq=40)


    # PRODUCTION
    print('RUNNING THE TEST FOR PRODUCTION')

    X = [ident_prod.get_data(picks = 'mag', tmin=tmin, tmax=tmax), unrel_prod.get_data(picks = 'mag',tmin=tmin, tmax=tmax)]
    X = [np.transpose(x, (0, 2, 1)) for x in X]

    adjacency, ch_names = find_ch_adjacency(epochs_prod.info, ch_type = 'mag')

    #RUNNING THE TEST
    tail = 1
    alpha_cluster_forming = 0.001
    n_conditions = 2  # was 4
    n_observations = len(X[0])
    dfn = n_conditions - 1
    dfd = n_observations - n_conditions

    f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)
    cluster_stats = spatio_temporal_cluster_test(X, n_permutations=1000, threshold=f_thresh, tail=tail,
                                                 adjacency=adjacency) #, step_down_p=0.1

    F_obs, clusters, p_values, _ = cluster_stats
    n_clusters_prod = len(clusters)
    print(p_values)
    p_accept = 0.2
    good_cluster_inds = np.where(p_values < p_accept)[0]
    print(good_cluster_inds)

    evokeds = {"Production Identical": ident_prod.average(), "Production Unrelated": unrel_prod.average()}

    for i_clu, clu_idx in enumerate(good_cluster_inds):
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        f_map = F_obs[time_inds, ...].mean(axis=0)

        sig_times = epochs_prod.times[time_inds]

        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True #ADDED

        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout='constrained')
        epochs_prod_temp = epochs_prod.pick(picks='mag')
        f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs_prod_temp.info, tmin=0)
        f_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            cmap='Reds',
            vlim=(np.min, np.max),
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10) #ADDED
        )
        image = ax_topo.images[0]

        ax_topo.set_title("")

        divider = make_axes_locatable(ax_topo)

        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            "Averaged F-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
        )
        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes("right", size="300%", pad=1.2)
        title = f"Cluster #{i_clu + 1}, {len(ch_inds)} sensor"
        if len(ch_inds) > 1:
            title += "s (mean) with p value " + str(p_values[clu_idx])
        plot_compare_evokeds(
            evokeds,
            title=title,
            picks=ch_inds,
            axes=ax_signals,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
        )
        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx(
            (ymin, ymax), sig_times[0]+0.6, sig_times[-1]+0.6, color="orange", alpha=0.3
        )
        fname = '/Users/admin/Box Sync/Starling/Experiment1/Figures/Sensor space single subject/MINMAX_2/' + sub + '_production_tminmax__' + str(i_clu+1)
        fig.savefig(fname)
        # plt.show()
    #ADD AUTOSAVING THE PLOT HERE
    #COMPREHENSION

    print("RUNNING THE TEST FOR COMPREHENSION")

    X = [ident_comp.get_data(picks='mag', tmin=tmin, tmax=tmax), unrel_comp.get_data(picks='mag',tmin=tmin, tmax=tmax)]
    X = [np.transpose(x, (0, 2, 1)) for x in X]

    adjacency, ch_names = find_ch_adjacency(epochs_comp.info, ch_type='mag')

    # RUNNING THE TEST
    tail = 1
    alpha_cluster_forming = 0.001
    n_conditions = 2  # was 4
    n_observations = len(X[0])
    dfn = n_conditions - 1
    dfd = n_observations - n_conditions

    f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)
    cluster_stats = spatio_temporal_cluster_test(X, n_permutations=1000, threshold=f_thresh, tail=tail,
                                                 adjacency=adjacency)  # , step_down_p=0.1

    F_obs, clusters, p_values, _ = cluster_stats
    n_clusters_comp = len(clusters)
    print(p_values)
    p_accept = 0.2
    good_cluster_inds = np.where(p_values < p_accept)[0]
    print(good_cluster_inds)

    evokeds = {"Comprehension Identical": ident_comp.average(), "Comprehension Unrelated": unrel_comp.average()}

    for i_clu, clu_idx in enumerate(good_cluster_inds):
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        f_map = F_obs[time_inds, ...].mean(axis=0)

        sig_times = epochs_comp.times[time_inds]

        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True  # ADDED

        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout='constrained')
        epochs_comp_temp = epochs_prod.pick(picks='mag')
        f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs_comp_temp.info, tmin=0)
        f_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            cmap='Reds',
            vlim=(np.min, np.max),
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10)  # ADDED
        )
        image = ax_topo.images[0]

        ax_topo.set_title("")

        divider = make_axes_locatable(ax_topo)

        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            "Averaged F-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
        )
        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes("right", size="300%", pad=1.2)
        title = f"Cluster #{i_clu + 1}, {len(ch_inds)} sensor"
        if len(ch_inds) > 1:
            title += "s (mean) with p value " + str(p_values[clu_idx])
        plot_compare_evokeds(
            evokeds,
            title=title,
            picks=ch_inds,
            axes=ax_signals,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
        )
        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx(
            (ymin, ymax), sig_times[0]+0.6, sig_times[-1]+0.6, color="orange", alpha=0.3
        )
        fname = '/Users/admin/Box Sync/Starling/Experiment1/Figures/Sensor space single subject/MINMAX_2/' + sub + '_comprehension_tminmax__' + str(i_clu + 1)
        fig.savefig(fname)
        # plt.show()
    return n_clusters_comp, n_clusters_prod
    # mne.viz.plot_compare_evokeds([ident_prod.average(), unrel_prod.average(), ident_comp.average(), unrel_comp.average()])


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Statistical testing for a single subject")
    # parser.add_argument("sub", type=str, help="Subject number (e.g., 101)")
    results = []
    # args = parser.parse_args()
    subs = ['R3250', 'R3254', 'R3261', 'R3264', 'R3270','R3271','R3272','R3273','R3275','R3277','R3279','R3285','R3286','R3289','R3290']

    for sub in subs:
        comp, prod = statistics_single_sensor(sub)
        results.append([sub, prod, comp])
    # df = pd.DataFrame(results, columns =['Subject', 'ProdClusters', 'CompClusters'])
    # output_file = '/Users/admin/Box Sync/Starling/Experiment1/Figures/Sensor space single subject/numClu.xlsx'
    # df.to_excel(output_file, index=False)

