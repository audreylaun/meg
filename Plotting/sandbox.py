import mne
import matplotlib.pyplot as plt
import numpy as np

def epoch_inspect(sub):
    directory = '/Users/admin/Box Sync/Starling/Experiment1/MEG_data/' + sub + '/'
    fname = directory + sub + '_comp-epo.fif'

    epochs = mne.read_epochs(fname, preload = True)

    print(epochs.info)

    for i in range(len(epochs['comprehension identical'])):
        trial = epochs['comprehension identical'][i].average().filter(l_freq=0, h_freq=20)
        mne.viz.plot_compare_evokeds(trial, picks = 'mag')

def sanity_check(sub):
    subjects_dir = '/Applications/freesurfer/8.0.0/subjects'
    directory = f'/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/{sub}/'

    stcs = {
        'Comprehension identical': mne.read_source_estimate(f"{directory}{sub}_ident_comp-lh.stc", sub),
        'Comprehension unrelated': mne.read_source_estimate(f"{directory}{sub}_unrel_comp-lh.stc", sub),
        'Production identical': mne.read_source_estimate(f"{directory}{sub}_ident_prod-lh.stc", sub),
        'Production unrelated': mne.read_source_estimate(f"{directory}{sub}_unrel_prod-lh.stc", sub),
    }

    brains = []

    cmaps = ['hot', 'cool', 'spring', 'winter']
    for (title, stc), cmap in zip(stcs.items(), cmaps):
        brain = stc.plot(
            subjects_dir=subjects_dir,
            hemi='lh',
            colormap=cmap,
            clim=dict(kind="value", lims=[3, 6, 9]),
            smoothing_steps=7,
            title=title
        )
        brain.add_text(0.1, 0.9, title, "title", font_size=14)
        brains.append(brain)  # Save each brain object

    # Now block until all windows are closed
    from pyvistaqt import BackgroundPlotter
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance()
    if app is not None:
        app.exec_()  # Start the Qt event loop manually

def region_plotting(sub, choice):
    '''
    put choice = 'stg' if you want to only plot the superior temporal gyrus
    put anything else to plot the entire temporal lobe
    '''
    subjects_dir = '/Applications/freesurfer/7.4.1/subjects'
    directory = f'/Users/admin/Box Sync/Starling/Experiment1/STCs/{sub}/'

    # STC files
    stc_ident = mne.read_source_estimate(f'{directory}/{sub}_ident_comp-lh.stc', subject=sub)
    stc_unrel = mne.read_source_estimate(f'{directory}/{sub}_unrel_comp-lh.stc', subject=sub)

    # ---- Get label for the STG ----
    # Use 'aparc' for Desikan-Killiany (common) or 'aparc.a2009s' for finer resolution
    labels = mne.read_labels_from_annot(subject=sub, parc='aparc', subjects_dir=subjects_dir)

    if choice == 'stg':
        final_labels = [label for label in labels if
                  'superiortemporal' in label.name and label.hemi == 'lh']  # left hemisphere STG
    elif choice == 'fusiform':
        final_labels = [label for label in labels if
                    'fusiform' in label.name and label.hemi == 'lh']  # left hemisphere STG
    else:
        # Entire temporal lobe
        temporal_keywords = [
           'superiortemporal', 'middletemporal', 'inferiortemporal', 'bankssts',
           'temporalpole', 'transversetemporal', 'entorhinal', 'parahippocampal', 'fusiform'
        ]

        # Filter only left hemisphere temporal labels
        final_labels = [
            label for label in labels
            if any(k in label.name for k in temporal_keywords) and label.hemi == 'lh'
        ]

    # ---- Extract mean activation in STG label ----
    src = mne.setup_source_space(sub, spacing="oct6", add_dist="patch", subjects_dir=subjects_dir)

    mean_ident = mne.extract_label_time_course(stc_ident, final_labels, src=src, mode='mean_flip')[0]
    mean_unrel = mne.extract_label_time_course(stc_unrel, final_labels, src=src, mode='mean_flip')[0]

    # ---- Plot ----
    times = stc_ident.times * 1000  # convert to ms

    plt.figure(figsize=(10, 5))
    plt.plot(times, mean_ident, label='Comprehension Identical', color='blue')
    plt.plot(times, mean_unrel, label='Comprehension Unrelated', color='red')
    plt.xlabel('Time (ms)')
    if choice == 'stg':
        region = "Superior Temporal Gyrus"
        plt.ylabel(f"Mean {region} Activation")
        plt.title(f"{region} Activation Across Conditions for {sub}")
    else:
        region = "Temporal Lobe"
        plt.ylabel(f"Mean {region} Activation")
        plt.title(f"{region} Activation Across Conditions for {sub}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def grand_average_sensor():
    # subs = ['R3250', 'R3254', 'R3261', 'R3264', 'R3270','R3271','R3272','R3273','R3275','R3277','R3279','R3285','R3286','R3289','R3290', 'R3326','R3327','R3328', 'R3329']
    subs = ['R3250', 'R3254', 'R3261', 'R3264', 'R3270','R3271','R3272','R3273','R3275','R3277','R3279','R3285','R3286','R3289','R3290','R3285','R3286','R3289','R3290','R3326','R3327','R3328', 'R3329']
    prod_ident = []
    prod_unrel = []
    comp_ident = []
    comp_unrel = []

    for sub in subs:
<<<<<<< HEAD
        directory = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/' + sub + '/'
=======
        directory = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/Testing/' + sub + '/'
>>>>>>> 99371e3f79805ce2c9605203223f22b6524c6a64
        fname = directory + sub + '_prod-epo.fif'
        prod = mne.read_epochs(fname, verbose=False)
        a = prod['production identical']
        b = prod['production unrelated']

        fname = directory + sub + '_comp-epo.fif'
        comp = mne.read_epochs(fname, verbose=False)
        c = comp['comprehension identical']
        d = comp['comprehension unrelated']

        mne.epochs.equalize_epoch_counts([a,b,c,d])

        prod_ident_evoked = a.average().filter(l_freq=0, h_freq=40)
        prod_unrel_evoked = b.average().filter(l_freq=0, h_freq=40)
        comp_ident_evoked = c.average().filter(l_freq=0, h_freq=40)
        comp_unrel_evoked = d.average().filter(l_freq=0, h_freq=40)

        prod_ident.append(prod_ident_evoked)
        prod_unrel.append(prod_unrel_evoked)
        comp_ident.append(comp_ident_evoked)
        comp_unrel.append(comp_unrel_evoked)

        print(len(a))
        print(len(b))
        print(len(c))
        print(len(d))

    prod_ident_all = mne.grand_average(prod_ident)
    prod_unrel_all = mne.grand_average(prod_unrel)
    comp_ident_all = mne.grand_average(comp_ident)
    comp_unrel_all = mne.grand_average(comp_unrel)

    mne.viz.plot_compare_evokeds([prod_ident_all, prod_unrel_all, comp_ident_all, comp_unrel_all])

    return prod_ident_all, prod_unrel_all, comp_ident_all, comp_unrel_all

def auto_reject(sub):
    directory = '/Users/admin/Box Sync/Starling/Experiment1/MEG_data/' + sub + '/'

    fname = directory + '/' + sub + '_comp_TEST-epo.fif'
    epochs_comp = mne.read_epochs(fname, preload=True)
    print(epochs_comp.info)

    fname = directory + '/' + sub + '_prod_TEST-epo.fif'
    epochs_prod = mne.read_epochs(fname, preload=True)
    print(epochs_prod.info)

    ident_prod = epochs_prod['production identical']
    unrel_prod = epochs_prod['production unrelated']
    ident_comp = epochs_comp['comprehension identical']
    unrel_comp = epochs_comp['comprehension unrelated']

    mne.epochs.equalize_epoch_counts([ident_prod, unrel_prod, ident_comp, unrel_comp], method='random', random_state=1)

    ident_prod_evk = ident_prod.average().filter(l_freq=None, h_freq=20)
    unrel_prod_evk = unrel_prod.average().filter(l_freq=None, h_freq=20)
    ident_comp_evk = ident_comp.average().filter(l_freq=None, h_freq=20)
    unrel_comp_evk = unrel_comp.average().filter(l_freq=None, h_freq=20)

    mne.viz.plot_compare_evokeds([ident_prod_evk, unrel_prod_evk, ident_comp_evk, unrel_comp_evk], picks='mag', title='NO AUTOREJECT')

    reject_criteria = dict(mag=3000e-15) #I NEED TO MAKE SURE THIS IS PICOTESLAS?
    epochs_prod.drop_bad(reject=reject_criteria)
    print(epochs_prod.drop_log)
    epochs_comp.drop_bad(reject=reject_criteria)
    print(epochs_comp.drop_log)

    ident_prod = epochs_prod['production identical']
    unrel_prod = epochs_prod['production unrelated']
    ident_comp = epochs_comp['comprehension identical']
    unrel_comp = epochs_comp['comprehension unrelated']

    mne.epochs.equalize_epoch_counts([ident_prod, unrel_prod, ident_comp, unrel_comp], method='random', random_state=1)

    ident_prod_evk = ident_prod.average().filter(l_freq=None, h_freq=20)
    unrel_prod_evk = unrel_prod.average().filter(l_freq=None, h_freq=20)
    ident_comp_evk = ident_comp.average().filter(l_freq=None, h_freq=20)
    unrel_comp_evk = unrel_comp.average().filter(l_freq=None, h_freq=20)

    mne.viz.plot_compare_evokeds([ident_prod_evk, unrel_prod_evk, ident_comp_evk, unrel_comp_evk], picks='mag', title='BAD AUTOREJECTED')

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

def get_stcs(subjects_dir, fsave_vertices, directory, sub):
    src_fname = '/Applications/freesurfer/8.0.0/subjects/fsaverage/bem/fsaverage-ico-4-src.fif'
    src = mne.read_source_spaces(src_fname)

    stc_prod_ident = mne.read_source_estimate(directory + sub + '_ident_prod-lh.stc', subject=sub)
    stc_prod_unrel = mne.read_source_estimate(directory + sub + '_unrel_prod-lh.stc', subject=sub)
    stc_comp_ident = mne.read_source_estimate(directory + sub + '_ident_comp-lh.stc', subject=sub)
    stc_comp_unrel = mne.read_source_estimate(directory + sub + '_unrel_comp-lh.stc', subject=sub)

    morph1 = mne.compute_source_morph(
        stc_prod_ident,
        subject_from=sub,
        subject_to='fsaverage',
        src_to=src,
        subjects_dir=subjects_dir)
    morph2 = mne.compute_source_morph(
        stc_prod_unrel,
        subject_from=sub,
        subject_to='fsaverage',
        src_to = src,
        subjects_dir=subjects_dir)
    morph3 = mne.compute_source_morph(
        stc_comp_ident,
        subject_from=sub,
        subject_to='fsaverage',
        src_to = src,
        subjects_dir=subjects_dir)
    morph4 = mne.compute_source_morph(
        stc_comp_unrel,
        subject_from=sub,
        subject_to='fsaverage',
        src_to = src,
        subjects_dir=subjects_dir)

    prod_ident = morph1.apply(stc_prod_ident)
    prod_unrel = morph2.apply(stc_prod_unrel)
    comp_ident = morph3.apply(stc_comp_ident)
    comp_unrel = morph4.apply(stc_comp_unrel)

    return prod_ident, prod_unrel, comp_ident, comp_unrel

def plot_stcs():
    subs = ['R3250', 'R3254', 'R3261', 'R3264', 'R3270','R3271','R3272','R3273','R3275','R3277','R3279','R3285','R3286','R3289','R3290']

    n_subjects = len(subs)
    subjects_dir = '/Applications/freesurfer/8.0.0/subjects'
    src_fname = '/Applications/freesurfer/8.0.0/subjects/fsaverage/bem/fsaverage-ico-4-src.fif'

    # arrays where the stcs for each condition will be stored
    ident_prod_stcs = []
    unrel_prod_stcs = []
    ident_comp_stcs = []
    unrel_comp_stcs = []

    # Reading the source space for fsaverage
    src = mne.read_source_spaces(src_fname)
    fsave_vertices = [s["vertno"] for s in src]

    for sub in subs:
        directory = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/' + sub + '/'
        stc_prod_ident_fsavg, stc_prod_unrel_fsavg, stc_comp_ident_fsavg, stc_comp_unrel_fsavg = get_stcs(subjects_dir,
                                                                                                          fsave_vertices, directory, sub)
        tstep = stc_prod_ident_fsavg.tstep * 1000

        ident_prod_stcs.append(stc_prod_ident_fsavg)
        unrel_prod_stcs.append(stc_prod_unrel_fsavg)
        ident_comp_stcs.append(stc_comp_ident_fsavg)
        unrel_comp_stcs.append(stc_comp_unrel_fsavg)

    def grand_average_stcs(stcs):
        """Average a list of STC objects."""
        data = np.stack([stc.data for stc in stcs], axis=0)  # (n_subjects, n_vertices, n_times)
        avg_data = np.mean(data, axis=0)
        return mne.SourceEstimate(
            avg_data,
            vertices=stcs[0].vertices,
            tmin=stcs[0].tmin,
            tstep=stcs[0].tstep,
            subject=stcs[0].subject
        )

    avg_ident_prod = grand_average_stcs(ident_prod_stcs)
    avg_unrel_prod = grand_average_stcs(unrel_prod_stcs)
    avg_ident_comp = grand_average_stcs(ident_comp_stcs)
    avg_unrel_comp = grand_average_stcs(unrel_comp_stcs)

    # === Step 2: Read temporal lobe label ===
    labels = mne.read_labels_from_annot(
        'fsaverage', parc='aparc.a2009s', hemi='lh', subjects_dir=subjects_dir
    )

    temporal_labels = [
        'G_temp_sup-G_T_transv-lh', 'G_temp_sup-Lateral-lh', 'G_temp_sup-Plan_polar-lh',
        'G_temp_sup-Plan_tempo-lh', 'G_temporal_inf-lh', 'G_temporal_middle-lh',
        'Pole_temporal-lh', 'S_temporal_inf-lh', 'S_temporal_sup-lh', 'S_temporal_transverse-lh'
    ]

    frontal_labels = [
    "G_and_S_frontomargin-lh",
    "G_and_S_precentral-lh",
    "G_front_middle-lh",
    "G_front_sup-lh",
    "G_opercularis-lh",
    "G_orbital-lh",
    "G_subcentral-lh",
    "G_suborbital-lh",
    "Pole_frontal-lh",
    "S_front_inf-lh",
    "S_front_middle-lh",
    "S_orbital_lateral-lh",
    "S_orbital_med-olfact-lh",
    "S_orbital-H_Shaped-lh",
    "S_precentral-inf-part-lh",
    "S_precentral-sup-part-lh",
    "S_subcentral_ant-lh",
    "S_subcentral_post-lh",
    "S_front_sup-lh"
]

    # Combine into a single label
    from functools import reduce
    import operator

    label_now = frontal_labels

    label_now_now = reduce(operator.add, [label for label in labels if label.name in label_now])

    # === Step 3: Process all 4 conditions ===
    condition_stcs = {
        'Ident-Prod': ident_prod_stcs,
        'Unrel-Prod': unrel_prod_stcs,
        'Ident-Comp': ident_comp_stcs,
        'Unrel-Comp': unrel_comp_stcs
    }

    timecourses = {}
    for label, stcs in condition_stcs.items():
        avg_stc = grand_average_stcs(stcs)
        # temporal_stc = avg_stc.in_label(label_now_now)
        temporal_stc = avg_stc
        timecourses[label] = temporal_stc.data.mean(axis=0)  # average across vertices
        times = temporal_stc.times * 1000  # convert to ms (only need once)

    # === Step 4: Plot all conditions ===
    plt.figure(figsize=(12, 5))

    for label, timecourse in timecourses.items():
        plt.plot(times, abs(timecourse * 1e9), label=label)  # convert Am to nAm

    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (nAm)')
    plt.title('Grand-Averaged Activation')
    plt.axvline(0, color='k', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    difference_prod = avg_unrel_prod - avg_ident_prod
    difference_comp = avg_unrel_comp - avg_ident_comp

    stcs = {
        'Comprehension difference': difference_comp,
        'Production difference': difference_prod
    }

    # stcs = {
    #     'Comprehension identical': avg_ident_comp,
    #     'Comprehension unrelated': avg_unrel_comp,
    #     'Production identical': avg_ident_prod,
    #     'Production unrelated': avg_unrel_prod,
    # }

    brains = []

    cmaps = ['hot', 'cool', 'spring', 'winter']
    for (title, stc), cmap in zip(stcs.items(), cmaps):
        brain = stc.plot(
            subjects_dir=subjects_dir,
            hemi='lh',
            colormap=cmap,
            clim=dict(kind="value", lims=[3, 6, 9]),
            smoothing_steps=7,
            title=title
        )
        brain.add_text(0.1, 0.9, title, "title", font_size=14)
        brains.append(brain)  # Save each brain object

    # Now block until all windows are closed
    from pyvistaqt import BackgroundPlotter
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance()
    if app is not None:
        app.exec_()  # Start the Qt event loop manually

def num_epochs():
    subs = ['R3250', 'R3254', 'R3261', 'R3264', 'R3270','R3271','R3272','R3273','R3275','R3277','R3279','R3285','R3286','R3289','R3290']

    for sub in subs:
        directory = '/Users/admin/Box Sync/Starling/Experiment1/MEG_data/' + sub + '/'
        fname = directory + sub + '_prod-epo.fif'
        prod = mne.read_epochs(fname, verbose=False)
        a = prod['production identical']
        b = prod['production unrelated']
        print(sub)
        print(len(a))
        print(len(b))
        reject_criteria = dict(mag=3000e-15)  # I NEED TO MAKE SURE THIS IS PICOTESLAS?
        prod.drop_bad(reject=reject_criteria)
        a = prod['production identical']
        b = prod['production unrelated']
        print(len(a))
        print(len(b))

def evoked_check(sub):
<<<<<<< HEAD
    directory = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/' + sub + '/'
=======
    directory = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/Testing/' + sub + '/'
>>>>>>> 99371e3f79805ce2c9605203223f22b6524c6a64
    fname = directory + sub + '_prod-epo.fif'
    prod = mne.read_epochs(fname, verbose=False)
    a = prod['production identical']
    b = prod['production unrelated']

    fname = directory + sub + '_comp-epo.fif'
    comp = mne.read_epochs(fname, verbose=False)
    c = comp['comprehension identical']
    d = comp['comprehension unrelated']

    mne.epochs.equalize_epoch_counts([a, b, c, d])

    prod_ident_evoked = a.average().filter(l_freq=0, h_freq=40)
    prod_unrel_evoked = b.average().filter(l_freq=0, h_freq=40)
    comp_ident_evoked = c.average().filter(l_freq=0, h_freq=40)
    comp_unrel_evoked = d.average().filter(l_freq=0, h_freq=40)

<<<<<<< HEAD
    mne.viz.plot_compare_evokeds([prod_ident_evoked, prod_unrel_evoked, comp_ident_evoked, comp_unrel_evoked])

# evoked_check('R3273')
plot_stcs()
=======
    mne.viz.plot_compare_evokeds([prod_ident_evoked, prod_unrel_evoked, comp_ident_evoked, comp_unrel_evoked], picks='mag')

# for i in ['R3250','R3254','R3261','R3264','R3270','R3271','R3272','R3273','R3275','R3277','R3279','R3285','R3326', 'R3327', 'R3328', 'R3329']:
#     evoked_check(i)
# plot_stcs()
>>>>>>> 99371e3f79805ce2c9605203223f22b6524c6a64
