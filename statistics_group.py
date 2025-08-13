import numpy as np
from numpy.random import randn
from scipy import stats as stats
import mne
import matplotlib.pyplot as plt
import argparse
from mne.channels import find_ch_adjacency
from mne.stats import combine_adjacency, spatio_temporal_cluster_test
from mne.viz import plot_compare_evokeds
from mne.epochs import equalize_epoch_counts
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from mpl_toolkits.axes_grid1 import ImageGrid, inset_locator, make_axes_locatable



def statistics_sensor():
    subs = ['R3250', 'R3254', 'R3261', 'R3264', 'R3270','R3271','R3272','R3273','R3275','R3277','R3279','R3285','R3286','R3289','R3290']

    ident_prod_evokeds = []
    unrel_prod_evokeds = []
    ident_comp_evokeds = []
    unrel_comp_evokeds = []

    #Load the evoked data to arrays by condition
    for sub in subs:
        directory = '/Users/admin/Box Sync/Starling/Experiment1/MEG_data/' + sub + '/'
        tmin=0.3
        tmax=0.7

        #Load all epoched data
        prod_fname = directory + sub + '_prod_TEST-epo.fif'
        epochs_prod = mne.read_epochs(prod_fname, preload =True, verbose=False)
        comp_fname = directory + sub + '_comp_TEST-epo.fif'
        epochs_comp = mne.read_epochs(comp_fname, preload = True, verbose=False)

        ident_prod = epochs_prod['production identical']
        unrel_prod = epochs_prod['production unrelated']
        ident_comp = epochs_comp['comprehension identical']
        unrel_comp = epochs_comp['comprehension unrelated']

        mne.epochs.equalize_epoch_counts([ident_prod, unrel_prod, ident_comp, unrel_comp], method = 'random')

        ident_prod_evokeds.append(ident_prod.average(picks='mag').filter(l_freq=0, h_freq=40, verbose=False))
        unrel_prod_evokeds.append(unrel_prod.average(picks='mag').filter(l_freq=0, h_freq=40, verbose=False))
        ident_comp_evokeds.append(ident_comp.average(picks='mag').filter(l_freq=0, h_freq=40, verbose=False))
        unrel_comp_evokeds.append(unrel_comp.average(picks='mag').filter(l_freq=0, h_freq=40, verbose=False))


    #Grand average the evoked data (for plotting)
    ident_prod_ga = mne.grand_average(ident_prod_evokeds)
    unrel_prod_ga = mne.grand_average(unrel_prod_evokeds)
    ident_comp_ga = mne.grand_average(ident_comp_evokeds)
    unrel_comp_ga = mne.grand_average(unrel_comp_evokeds)

    # PRODUCTION
    #Conduct the test for production, with test only run for tmin to tmax time window
    prod_ident_data = np.stack([x.get_data(picks='mag', tmin=tmin, tmax=tmax) for x in ident_prod_evokeds], axis=0)
    prod_unrel_data = np.stack([x.get_data(picks='mag', tmin=tmin, tmax=tmax) for x in unrel_prod_evokeds], axis=0)

    del ident_prod_evokeds, unrel_prod_evokeds

    #X shape: (# of conditions, # of trials/subjects, # of timepoints, # of channels)
    X = np.stack([prod_ident_data, prod_unrel_data])
    X = [np.transpose(x, (0,2,1)) for x in X]

    adjacency, ch_names = find_ch_adjacency(ident_prod_ga.info, ch_type = 'mag')

    # RUNNING THE TEST
    #Set parameters
    tail = 1
    alpha_cluster_forming = 0.1
    n_conditions = 2
    n_observations = len(X[0])
    dfn = n_conditions - 1
    dfd = n_observations - n_conditions
    f_thresh = stats.f.ppf(1-alpha_cluster_forming, dfn=dfn, dfd=dfd)

    #Run test
    cluster_stats = spatio_temporal_cluster_test(X, n_permutations=1000, threshold=f_thresh, tail=tail,
                                                 adjacency=adjacency) #, step_down_p=0.1
    F_obs, clusters, p_values, _ = cluster_stats
    print(p_values)
    p_accept = 0.4
    good_cluster_inds = np.where(p_values < p_accept)[0]
    print(good_cluster_inds)

    evokeds = {"Production Identical": ident_prod_ga, "Production Unrelated": unrel_prod_ga}

    #Plot "significant" clusters (< p_accept)
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        #obtain indeces
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)
        f_map = F_obs[time_inds, ...].mean(axis=0)
        sig_times = ident_prod_ga.times[time_inds + 600]

        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True  # ADDED

        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout='constrained')
        # epochs_prod_temp = ident_prod_ga.pick(picks='mag')

        #Plot topographic map of F values
        f_evoked = mne.EvokedArray(f_map[:, np.newaxis], ident_prod_ga.pick(picks='mag').info, tmin=0)
        f_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            cmap='Reds',
            vlim=(np.min, np.max),
            show=False,
            colorbar=False,
            mask_params = dict(markersize=10)  # ADDED
        )
        image = ax_topo.images[0]
        ax_topo.set_title("")
        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)
        # add axes for colorbar
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            "Averaged F-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
        )

        #Plot timecourse of significant cluster
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
            (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
        )
    plt.show()


    # COMPREHENSION

    comp_ident_data = np.stack([x.get_data(picks='mag',tmin=tmin, tmax=tmax) for x in ident_comp_evokeds], axis = 0)
    comp_unrel_data = np.stack([x.get_data(picks='mag',tmin=tmin, tmax=tmax) for x in unrel_comp_evokeds], axis = 0)

    del ident_comp_evokeds, unrel_comp_evokeds

    X = np.stack([comp_ident_data, comp_unrel_data])
    X = [np.transpose(x, (0,2,1)) for x in X]

    del comp_ident_data, comp_unrel_data

    # comp_fname = '/Users/admin/Box Sync/Starling/Experiment1/MEG_data/R3250/R3250_comp_TEST-epo.fif'
    # sample_comp = mne.read_epochs(comp_fname, preload=True)
    adjacency, ch_names = find_ch_adjacency(ident_comp_ga.info, ch_type='mag')

    # RUNNING THE TEST

    tail = 1
    alpha_cluster_forming = 0.1
    n_conditions = 2  # was 4
    n_observations = len(X[0])
    dfn = n_conditions - 1
    dfd = n_observations - n_conditions

    f_thresh = stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

    cluster_stats = spatio_temporal_cluster_test(X, n_permutations=1000, threshold=f_thresh, tail=tail,
                                                 adjacency=adjacency)

    F_obs, clusters, p_values, _ = cluster_stats
    print(p_values)
    p_accept = 0.4
    good_cluster_inds = np.where(p_values < p_accept)[0]
    print(good_cluster_inds)
    evokeds = {"Comprehension Identical": ident_comp_ga, "Comprehension Unrelated": unrel_comp_ga}

    for i_clu, clu_idx in enumerate(good_cluster_inds):
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        f_map = F_obs[time_inds, ...].mean(axis=0)

        sig_times = ident_comp_ga.times[time_inds + 600]

        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True  # ADDED

        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout='constrained')

        epochs_comp_temp = ident_comp_ga.pick(picks='mag')
        f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs_comp_temp.info, tmin=0)
        f_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            cmap='Reds',
            vlim=(np.min, np.max),
            show=False,
            colorbar=False
        )
        image = ax_topo.images[0]
        ax_topo.set_title("")
        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)
        # add axes for colorbar
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
            (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
        )
    plt.show()

def get_stcs(subjects_dir, fsave_vertices, directory, sub):
    stc_prod_ident = mne.read_source_estimate(directory + sub + '_ident_prod-lh.stc', subject=sub)
    stc_prod_unrel = mne.read_source_estimate(directory + sub + '_unrel_prod-lh.stc', subject=sub)
    stc_comp_ident = mne.read_source_estimate(directory + sub + '_ident_comp-lh.stc', subject=sub)
    stc_comp_unrel = mne.read_source_estimate(directory + sub + '_unrel_comp-lh.stc', subject=sub)

    morph = mne.compute_source_morph(
        src=mne.setup_source_space(sub, spacing="oct6", add_dist="patch", subjects_dir=subjects_dir),
        subject_from=sub,
        subject_to='fsaverage',
        spacing=fsave_vertices,
        subjects_dir=subjects_dir)
    prod_ident = morph.apply(stc_prod_ident)
    prod_unrel = morph.apply(stc_prod_unrel)
    comp_ident = morph.apply(stc_comp_ident)
    comp_unrel = morph.apply(stc_comp_unrel)

    return prod_ident, prod_unrel, comp_ident, comp_unrel

def statistics_source():
    '''
    Runs a cluster based permutation test on source data
    '''

    subs = ['R3250', 'R3254', 'R3261', 'R3264', 'R3270','R3271','R3272','R3273','R3275','R3277','R3279','R3285','R3286','R3289','R3290']

    n_subjects = len(subs)
    subjects_dir = '/Applications/freesurfer/7.4.1/subjects'
    src_fname = '/Applications/freesurfer/7.4.1/subjects/fsaverage/bem/fsaverage-ico-4-src.fif'

    # arrays where the stcs for each condition will be stored
    ident_prod_stcs = []
    unrel_prod_stcs = []
    ident_comp_stcs = []
    unrel_comp_stcs = []

    # Reading the source space for fsaverage
    src = mne.read_source_spaces(src_fname)
    fsave_vertices = [s["vertno"] for s in src]

    for sub in subs:
        directory = '/Users/admin/Box Sync/Starling/Experiment1/MEG_Data/' + sub + '/'
        stc_prod_ident_fsavg, stc_prod_unrel_fsavg, stc_comp_ident_fsavg, stc_comp_unrel_fsavg = get_stcs(subjects_dir,
                                                                                                          fsave_vertices, directory, sub)
        tstep = stc_prod_ident_fsavg.tstep * 1000

        ident_prod_stcs.append(stc_prod_ident_fsavg)
        unrel_prod_stcs.append(stc_prod_unrel_fsavg)
        ident_comp_stcs.append(stc_comp_ident_fsavg)
        unrel_comp_stcs.append(stc_comp_unrel_fsavg)

    # Generates two arrays of all stcs for each production condition
    prod_ident_data = np.stack([x.data for x in ident_prod_stcs], axis=0)
    prod_unrel_data = np.stack([x.data for x in unrel_prod_stcs], axis=0)

    # stacks the arrays and puts them into the proper configuration for the statistical testing
    X = np.stack([prod_ident_data, prod_unrel_data])
    X = np.transpose(X, (2, 3, 1, 0))

    X = np.abs(X)  # only magnitude
    X = X[:, :, :, 0] - X[:, :, :, 1]  # make paired contrast

    adjacency = mne.spatial_src_adjacency(src)

    # Note that X needs to be a multi-dimensional array of shape
    # observations (subjects) × time × space, so we permute dimensions
    X = np.transpose(X, [2, 1, 0])

    # Here we set a cluster forming threshold based on a p-value for
    # the cluster based permutation test.
    # We use a two-tailed threshold, the "1 - p_threshold" is needed
    # because for two-tailed tests we must specify a positive threshold.
    p_threshold = 0.001
    df = n_subjects - 1  # degrees of freedom for the test
    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

    # Now let's actually do the clustering. This can take a long time...
    print("Clustering.")
    T_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_1samp_test(
        X,
        adjacency=adjacency,
        n_jobs=None,
        threshold=t_threshold,
        buffer_size=None,
        verbose=True,
    )

    # Select the clusters that are statistically significant at p < 0.05
    good_clusters_idx = np.where(cluster_p_values < 0.5)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]

    print(i for i in cluster_p_values)

    print("Visualizing clusters.")

    # Now let's build a convenient representation of our results, where consecutive
    # cluster spatial maps are stacked in the time dimension of a SourceEstimate
    # object. This way by moving through the time dimension we will be able to see
    # subsequent cluster maps.
    stc_all_cluster_vis = summarize_clusters_stc(
        clu, tstep=tstep, vertices=fsave_vertices, subject="fsaverage"
    )

    # Let's actually plot the first "time point" in the SourceEstimate, which
    # shows all the clusters, weighted by duration.

    # blue blobs are for condition A < condition B, red for A > B
    brain = stc_all_cluster_vis.plot(
        hemi="both",
        views="lateral",
        subjects_dir=subjects_dir,
        time_label="temporal extent (ms)",
        size=(800, 800),
        smoothing_steps=5,
        clim=dict(kind="value", pos_lims=[0, 1, 40]),
    )

    # We could save this via the following:
    # brain.save_image('clusters.png')

def statistics_source_withrois():
    import numpy as np
    import mne
    from scipy import stats
    from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc

    def statistics_source():
        subs = ['R3250', 'R3254', 'R3261', 'R3264', 'R3270', 'R3271', 'R3272', 'R3273', 'R3275', 'R3277', 'R3279',
                'R3285', 'R3286', 'R3289', 'R3290']
        n_subjects = len(subs)
        subjects_dir = '/Applications/freesurfer/7.4.1/subjects'
        src_fname = '/Applications/freesurfer/7.4.1/subjects/fsaverage/bem/fsaverage-ico-4-src.fif'

        ident_prod_stcs = []
        unrel_prod_stcs = []
        ident_comp_stcs = []
        unrel_comp_stcs = []

        src = mne.read_source_spaces(src_fname)
        fsave_vertices = [s["vertno"] for s in src]

        # Define your ROIs (make sure names match exactly the FreeSurfer aparc labels)
        language_rois = [
            "inferiortemporal-lh", "middletemporal-lh", "superiortemporal-lh", "transversetemporal-lh",
            "temporalpole-lh", "insula-lh", "bankssts-lh", "fusiform-lh",
            "parsopercularis-lh", "parsorbitalis-lh", "parstriangularis-lh",
            "frontalpole-lh", "lateralorbitofrontal-lh", "medialorbitofrontal-lh", "rostralmiddlefrontal-lh",
            "inferiorparietal-lh", "supramarginal-lh"
        ]

        # Load left hemisphere labels from fsaverage aparc
        labels = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=subjects_dir, hemi="lh")
        roi_labels = [lab for lab in labels if lab.name in language_rois]
        roi_vertices = np.concatenate([lab.vertices for lab in roi_labels])

        # Create a boolean mask over left hemisphere vertices: True if vertex in ROI
        lh_vertices = fsave_vertices[0]
        vertex_mask = np.isin(lh_vertices, roi_vertices)

        for sub in subs:
            directory = f'/Users/admin/Box Sync/Starling/Experiment1/MEG_Data/{sub}/'
            stc_prod_ident_fsavg, stc_prod_unrel_fsavg, stc_comp_ident_fsavg, stc_comp_unrel_fsavg = get_stcs(
                subjects_dir, fsave_vertices, directory, sub)
            tstep = stc_prod_ident_fsavg.tstep * 1000  # ms

            ident_prod_stcs.append(stc_prod_ident_fsavg)
            unrel_prod_stcs.append(stc_prod_unrel_fsavg)
            ident_comp_stcs.append(stc_comp_ident_fsavg)
            unrel_comp_stcs.append(stc_comp_unrel_fsavg)

        prod_ident_data = np.stack([x.data for x in ident_prod_stcs], axis=0)  # shape: subjects x vertices x times
        prod_unrel_data = np.stack([x.data for x in unrel_prod_stcs], axis=0)

        # Stack conditions, reorder axes: subjects, times, vertices, conditions
        X = np.stack([prod_ident_data, prod_unrel_data])  # (2 conditions, subjects, vertices, times)
        X = np.transpose(X, (1, 3, 2, 0))  # now: subjects x times x vertices x conditions

        # Use magnitude and compute difference between conditions
        X = np.abs(X)
        X = X[:, :, :, 1] - X[:, :, :, 0]  # difference: condition 1 - 0

        # Select time points between 200 and 600 ms
        times = ident_prod_stcs[0].times * 1000  # convert to ms
        time_mask = (times >= 200) & (times <= 600)
        X = X[:, time_mask, :]  # subjects x selected times x all vertices

        # Restrict to left hemisphere vertices only
        n_lh = len(fsave_vertices[0])  # 2562
        n_rh = len(fsave_vertices[1])  # 2562

        # Data currently has vertices for both hemispheres concatenated along last axis: 5124 vertices
        # Separate hemispheres:
        X_lh = X[:, :, :n_lh]  # left hemisphere data

        # Now apply your ROI vertex mask (length n_lh)
        X_lh_roi = X_lh[:, :, vertex_mask]

        # Prepare adjacency matrix
        adjacency = mne.spatial_src_adjacency(src)  # full adjacency 5124x5124
        adj_lh = adjacency[:n_lh, :n_lh]  # left hemisphere adjacency

        # Apply ROI mask to adjacency
        adjacency_roi = adj_lh[vertex_mask, :][:, vertex_mask]

        # Cluster forming threshold
        p_threshold = 0.001
        df = n_subjects - 1
        t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

        print("Clustering.")
        T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
            X_lh_roi,
            adjacency=adjacency_roi,
            n_jobs=None,
            threshold=t_threshold,
            buffer_size=None,
            verbose=True,
        )

        # Select significant clusters at p<0.05
        print(cluster_p_values)
        good_clusters_idx = np.where(cluster_p_values < 0.05)[0]
        good_clusters = [clusters[idx] for idx in good_clusters_idx]

        print("Significant cluster p-values:", cluster_p_values[good_clusters_idx])

        # Visualize clusters
        # To visualize, create a SourceEstimate for the whole left hemisphere, setting zeros outside ROI
        # First create zeros array for lh vertices x times
        stc_data = np.zeros((len(lh_vertices), X_lh_roi.shape[1]))  # vertices x times

        # For each time, set cluster values from cluster mask (for visualization, here just combine all clusters)
        # This example just creates a sum mask of significant clusters:
        combined_mask = np.zeros((X_lh_roi.shape[1], len(vertex_mask)), dtype=bool)
        for clu in good_clusters:
            # clu is tuple (time_inds, vertex_inds) over reduced ROI vertices
            combined_mask[clu[0], clu[1]] = True

        # Sum cluster mask over time and space as a simple visualization example
        stc_data[vertex_mask, :] = combined_mask.T.astype(float)

        # Create SourceEstimate
        from mne import SourceEstimate
        stc_all_cluster_vis = SourceEstimate(
            stc_data,
            vertices=[lh_vertices, np.array([])],  # left hemisphere vertices, empty right hemi
            tmin=times[time_mask][0] / 1000,  # convert back to seconds
            tstep=tstep / 1000,
            subject='fsaverage'
        )

        brain = stc_all_cluster_vis.plot(
            hemi='lh',
            views='lateral',
            subjects_dir=subjects_dir,
            time_label="Temporal extent (ms)",
            size=(800, 800),
            smoothing_steps=5,
            clim=dict(kind="value", pos_lims=[0, 0.5, 1])
        )

    # We could save this via the following:
    # brain.save_image('clusters.png')

def eelbrain_stats():
    import os
    import eelbrain
    import mne
    import pickle
    import numpy as np
    subjects_dir = '/Applications/freesurfer/7.4.1/subjects'

    ROOT = '/Users/admin/Box Sync/Starling/Experiment1/MEG_data/'
    OUT = '/Users/admin/Box Sync/Starling/Experiment1/EELBRAIN_OUT'
    subjects = ['R3250', 'R3264', 'R3270','R3271','R3273','R3275','R3277','R3279','R3285','R3286','R3289','R3290']

    stcs = []
    prime_type = []
    task = []
    subject = []

    src_fname = '/Applications/freesurfer/7.4.1/subjects/fsaverage/bem/fsaverage-ico-4-src.fif'
    src = mne.read_source_spaces(src_fname)
    fsave_vertices = [s["vertno"] for s in src]

    language_rois = [
        "inferiortemporal-lh", "middletemporal-lh", "superiortemporal-lh", "transversetemporal-lh",
        "temporalpole-lh", "insula-lh", "bankssts-lh", "fusiform-lh",
        "parsopercularis-lh", "parsorbitalis-lh", "parstriangularis-lh",
        "frontalpole-lh", "lateralorbitofrontal-lh", "medialorbitofrontal-lh", "rostralmiddlefrontal-lh",
        "inferiorparietal-lh", "supramarginal-lh"
    ]

    labels = mne.read_labels_from_annot(
        'fsaverage', parc='aparc', hemi='lh', subjects_dir=subjects_dir
    )

    from functools import reduce
    import operator

    language_label = reduce(operator.add, [label for label in labels if label.name in language_rois])

    # language_label = sum([label for label in labels if label.name in language_rois])

    for subj in subjects:
        stc_path = os.path.join(ROOT, subj)
        files = [f for f in os.listdir(stc_path) if f.endswith('-lh.stc')]

        for f in files:
            stc_file = os.path.join(stc_path, f)
            stc = mne.read_source_estimate(stc_file)

            stc_temp = stc.in_label(language_label)

            morph = mne.compute_source_morph(
                stc_temp,  # input stc
                subject_from=subj,  # subject ID of this STC
                subject_to='fsaverage',  # target brain
                subjects_dir=subjects_dir,
                src_to=src,
                verbose=False
            )
            stc_fsavg = morph.apply(stc_temp)  # morph STC data

            stcs.append(stc_fsavg)

            parts = f.replace('-lh.stc', '').split('_')
            prime_type.append(parts[1])  # 'ident' or 'unrel'
            task.append(parts[2])  # 'comp' or 'prod'
            subject.append(parts[0])  # e.g., 'R3250'

            del stc

    # Create a new Eelbrain dataset
    ds = eelbrain.Dataset()

    # Load STC data into the dataset as an NDVar
    ds['stc'] = eelbrain.load.fiff.stc_ndvar(
        stcs,
        subject='fsaverage',
        src='ico-4',
        subjects_dir=subjects_dir,  # define this earlier in your script
        method='dSPM',
        fixed=False,
        parc='aparc'
    )

    # Add experimental factors
    ds['Task'] = eelbrain.Factor(task)  # 'comp' or 'prod'
    ds['Prime_Type'] = eelbrain.Factor(prime_type)  # 'ident' or 'unrel'
    ds['subject'] = eelbrain.Factor(subject, random=True)  # e.g., 'R3250'

    # Optional convenience alias for working with the source estimates
    src = ds['stc']

    # Conditions
    Tasks = ['prod', 'comp']
    Prime_Types = ['ident', 'unrel']
    pvalue = 0.05

    # ------------------ Cluster-based permutation test ------------------ #
    for current_task in Tasks:
        print(f'Running test for task: {current_task}')

        # Reset dataset to full source space
        # test = ds['stc']
        # source = test.source
        # roi_mask = source.parc.get_mask(language_rois)  # roi_mask is a boolean array matching vertices
        # stc_roi = test.sub(source=roi_mask)
        # ds['stc'] = stc_roi

        # Run paired samples t-test for current task
        res = eelbrain.testnd.TTestRelated(
            ds['stc'],              # x = NDVar (DV)
            ds['Prime_Type'],       # Y = within-subject condition
            match=ds['subject'],    # subject matching
            sub=ds['Task'] == current_task,
            pmin=0.05,
            tstart=0.3,
            tstop=0.6,
            samples=1000,
            mintime=0.01,
            minsource=5,
            parc=None,
        )

        print(res.clusters)

        # # Save full result object
        # os.makedirs(OUT + f'Results/{current_task}/', exist_ok=True)
        # with open(OUT + f'Results/{current_task}/res.p', 'wb') as f:
        #     pickle.dump(res, f)
        #
        # # Save cluster summary table
        # with open(OUT + f'Results/{current_task}/results_table.txt', 'w') as f:
        #     f.write(f'Model: {res.X}, N={len(subjects)}\n')
        #     f.write(f'tstart={res.tstart}, tstop={res.tstop}, samples={res.samples}, pmin={res.pmin}\n\n')
        #     f.write(str(res.clusters))
        #
        # # Identify significant clusters
        # ix_sign_clusters = np.where(res.clusters['p'] <= pvalue)[0]
        #
        # for i, clu_idx in enumerate(ix_sign_clusters):
        #     cluster = res.clusters[clu_idx]['cluster']
        #     tstart = res.clusters[clu_idx]['tstart']
        #     tstop = res.clusters[clu_idx]['tstop']
        #     effect = res.clusters[clu_idx]['effect']
        #
        #     # Rename for plotting compatibility
        #     if effect == 'Prime_Type':
        #         effect_label = 'Prime_Type'
        #     else:
        #         effect_label = effect  # no change, placeholder for future cases
        #
        #     # Save cluster as label for plotting
        #     label = eelbrain.labels_from_clusters(cluster)
        #     label[0].name = 'label-lh'
        #     mne.write_labels_to_annot(
        #         label,
        #         subject='fsaverage',
        #         parc=f'cluster{i}_FullAnalysis',
        #         subjects_dir=subjects_dir,
        #         overwrite=True
        #     )
        #
        #     # Subset source to cluster label
        #     src.source.set_parc(f'cluster{i}_FullAnalysis')
        #     src_region = src.sub(source='label-lh')
        #     ds['stc'] = src_region
        #     timecourse = src_region.mean('source')
        #
        #     # --- 1) Timecourse Plot ---
        #     activation = eelbrain.plot.UTSStat(
        #         timecourse,
        #         effect_label,
        #         ds=ds,
        #         sub=ds['Task'] == current_task,
        #         legend='lower left',
        #         title=f'cluster{i + 1} time course, Task={current_task}, effect={effect}'
        #     )
        #     activation.add_vspan(xmin=tstart, xmax=tstop, color='lightgrey', zorder=-50)
        #     activation.save(
        #         OUT + f'Results/{current_task}/cluster{i + 1}_timecourse_({tstart}-{tstop})_effect={effect}.png')
        #     activation.close()
        #
        #     # --- 2) Brain Plot ---
        #     brain = eelbrain.plot.brain.cluster(
        #         cluster.mean('time'),
        #         subjects_dir=subjects_dir,
        #         surf='smoothwm'
        #     )
        #     brain.save_image(
        #         OUT + f'Results/{current_task}/cluster{i + 1}_brain_({tstart}-{tstop})_Task={current_task}_effect={effect}.png')
        #     brain.close()
        #
        #     # --- 3) Bar Graph ---
        #     ds['average_source_activation'] = timecourse.mean(time=(tstart, tstop))
        #     bar = eelbrain.plot.Barplot(
        #         ds['average_source_activation'],
        #         X=effect_label,
        #         ds=ds,
        #         sub=ds['Task'] == current_task
        #     )
        #     bar.save(
        #         OUT + f'Results/{current_task}/cluster{i + 1}_BarGraph_({tstart}-{tstop})_Task={current_task}_effect={effect}.png')
        #     bar.close()

    return 'My boyfriend is stinky'

