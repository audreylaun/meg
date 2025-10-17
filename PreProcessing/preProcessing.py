import mne
import numpy as np
from matplotlib import pyplot as plt
from mne.preprocessing import ICA, corrmap
import argparse
import pandas as pd
import openpyxl
<<<<<<< HEAD
=======
import os
>>>>>>> 99371e3f79805ce2c9605203223f22b6524c6a64

def fix_56(raw):
    # Fix the location of MEG 056, necessary for all data collected at KIT-UMD MEG Lab
    loc = np.array([
        0.09603, -0.07437, 0.00905, -0.5447052, -0.83848277,
        0.01558496, 0., -0.01858388, -0.9998273, 0.8386276,
        -0.54461113, 0.01012274])
    index = raw.ch_names.index('MEG 056')
    raw.info['chs'][index]['loc'] = loc
    return raw

def preprocess_meg(sub, condition):
<<<<<<< HEAD
    '''
    sub = subject number
    condition = condition number (A,B,C,D,E,F,G,H)

    Will access the following files:
    _prod-raw.fif, _comp-raw.fif, emptyroom.sqd

    Will save the following files
    _prod_preproc-raw.fif: preprocessed continuous data for production block
    _comp_preproc-raw.fif: preprocessed continuous data for comprehension block
    -proj.fif: two magnetometer signal space projectors calculated from empty room data
    _prod-epo.fif: epoch object for production trials, with bad productions excluded ("production unrelated", "production identical")
    _comp-epo.fif: epoch object for comprehension trials, with no trials excluded ("comprehension unrelated", "comprehension identical")
    '''

    print(f"Preprocessing MEG data for subject {sub} and condition {condition}...")
    comp_condition = 'X'
    prod_condition = 'X'
    if condition == 'A':
        comp_condition = 'A1'
        prod_condition = 'B1'
    elif condition == 'B':
        comp_condition = 'B1'
        prod_condition = 'A1'
    elif condition == 'C':
        comp_condition = 'B1'
        prod_condition = 'A1'
    elif condition == 'D':
        comp_condition = 'A1'
        prod_condition = 'B1'
    elif condition == 'E':
        comp_condition = 'A2'
        prod_condition = 'B2'
    elif condition == 'F':
        comp_condition = 'B2'
        prod_condition = 'A2'
    elif condition == 'G':
        comp_condition = 'B2'
        prod_condition = 'A2'
    elif condition == 'H':
        comp_condition = 'A2'
        prod_condition = 'B2'

    directory = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/' + sub + '/'

    # Empty Room Projectors
    empty_room_fname = directory + '/DAQ/' + sub + '_emptyroom.sqd'
    empty_room_raw = mne.io.read_raw_kit(empty_room_fname, preload = True)

    empty_room_raw = fix_56(empty_room_raw)

    empty_room_raw.plot()
    bads_input = input('Enter bad channels, separated by comma (MEG O11,MEG 012): ')
    bad_channels = [ch.strip() for ch in bads_input.split(',') if ch.strip()]
    empty_room_raw.info['bads'] = bad_channels
    print(f"Marked bad channels: {bad_channels}")

    empty_room_projs = mne.compute_proj_raw(empty_room_raw, n_mag = 2)
    fname = directory + sub + '-proj.fif'
    mne.write_proj(fname, empty_room_projs, overwrite = True)

    del empty_room_raw

    # PRODUCTION
    prod_file = directory + sub + '_prod-raw.fif'
    raw = mne.io.read_raw_fif(prod_file, preload=True)
    raw.plot()

    bads_input = input('Enter bad channels separated by commas (e.g., MEG 011,MEG 012): ')
    bad_channels = [ch.strip() for ch in bads_input.split(',') if ch.strip()]

    # Mark bad channels in the raw object
    raw.info['bads'] = bad_channels
    print(f"Marked bad channels: {bad_channels}")

    # Fix location of channel 56
    raw = fix_56(raw)

    # Interpolate bad channels
    raw = raw.interpolate_bads(method='MNE')  # Interpolate bads

    # Filter the ICA at a high pass of 1kHz to remove low frequency drifts,
    raw_filt = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica = ICA(n_components=30, max_iter="auto")
    ica.fit(raw_filt)
    #ica
    ica.plot_sources(raw, show_scrollbars=True)
    ica.plot_components()

    plt.show(block=True)

    ica_input = input('Enter ICA components to be removed, separated by a comma: ')
    ica_excluded = [int(ch.strip()) for ch in ica_input.split(',') if ch.strip()]

    ica.exclude = ica_excluded

    ica.apply(raw)

    del ica, raw_filt

    raw.add_proj(empty_room_projs)

    fname = directory + sub + '_prod_preproc-raw.fif'
    mne.io.Raw.save(raw, fname, overwrite=True)

    events = mne.find_events(raw, stim_channel="STI 014")
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        event_dict = {
            "production identical": 162,
            "production unrelated": 164,
        }
    else:
        event_dict = {
            "production identical": 162,
            "production unrelated": 164,
            "ignore":168
        }
    epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict, preload=True)
    excel_fname = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/Behavioral_Data/Productions_' + prod_condition + '.xlsx'
    df = pd.read_excel(excel_fname)
    column_name = sub + ' Accuracy'
    bad_productions = df.index[df[column_name] == False].tolist()
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        if prod_condition == 'A1':
            drop_always = [1, 20, 34, 39, 41, 43, 45, 57, 62, 70, 78, 81, 82, 84, 86, 88, 91, 92, 93, 94, 95,
                           96, 97, 98, 99, 101, 102, 103, 104, 105, 109, 111, 113, 114, 115, 119, 120, 121, 122, 123,
                           125, 129, 131, 135, 136, 138, 139, 140, 142, 143]
        else:
            drop_always = [2, 14, 15, 19, 23, 24, 35, 37, 64, 72, 75, 77, 78, 79, 83, 84, 87, 88, 89, 90, 97, 98, 100,
                           101,102, 105, 108, 110, 113, 115, 117, 118, 119, 121, 126, 127, 132, 133, 135, 136, 137, 138,
                           139, 140, 142, 143, 144, 145, 146, 148]
        drop = np.unique(np.concatenate((bad_productions, drop_always)))
    else:
        drop = bad_productions
    epochs.drop(drop)

    fname = directory + '/' + sub + '_prod-epo.fif'
    epochs.save(fname, overwrite=True)

    del raw, epochs


    #COMPREHENSION
    comp_file = directory + sub + '_comp-raw.fif'
    raw = mne.io.read_raw_fif(comp_file, preload=True)
    raw.plot()

    bads_input = input('Enter bad channels separated by commas (e.g., MEG 011,MEG 012): ')
    bad_channels = [ch.strip() for ch in bads_input.split(',') if ch.strip()]

    # Mark bad channels in the raw object
    raw.info['bads'] = bad_channels
    print(f"Marked bad channels: {bad_channels}")

    # Fix the location of MEG 056 (always)
    loc = np.array([
        0.09603, -0.07437, 0.00905, -0.5447052, -0.83848277,
        0.01558496, 0., -0.01858388, -0.9998273, 0.8386276,
        -0.54461113, 0.01012274])
    index = raw.ch_names.index('MEG 056')
    raw.info['chs'][index]['loc'] = loc

    # Interpolate bad channels
    raw = raw.interpolate_bads(method='MNE')  # Interpolate bads

    # Filter the ICA at a high pass of 1kHz to remove low frequency drifts,
    # We will apply the ICA to the unfiltered data, though, so we keep a copy
    raw_filt = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica = ICA(n_components=30, max_iter="auto")
    ica.fit(raw_filt)

    ica.plot_sources(raw, show_scrollbars=True)
    ica.plot_components()

    plt.show(block=True)

    ica_input = input('Enter ICA components to be removed, separated by a comma: ')
    ica_excluded = [int(ch.strip()) for ch in ica_input.split(',') if ch.strip()]

    ica.exclude = ica_excluded

    ica.apply(raw)

    del ica, raw_filt

    raw.add_proj(empty_room_projs)

    fname = directory + '/' + sub + '_comp_preproc-raw.fif'
    mne.io.Raw.save(raw, fname, overwrite=True)

    events = mne.find_events(raw, stim_channel="STI 014")
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        event_dict = {
            "comprehension identical": 162,
            "comprehension unrelated": 164,
        }
    else:
        event_dict = {
            "comprehension identical": 162,
            "comprehension unrelated": 164,
            "ignore": 168
        }
    epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict, preload=True)
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        if comp_condition=='A1':
            drop_always = [1, 20, 34, 39, 41, 43, 45, 57, 62, 70, 78, 81, 82, 84, 86, 88, 91, 92, 93, 94, 95,
                           96, 97, 98, 99, 101, 102, 103, 104, 105, 109, 111, 113, 114, 115, 119, 120, 121, 122, 123,
                           125, 129, 131, 135, 136, 138, 139, 140, 142, 143]

        else:
            drop_always = [2, 14, 15, 19, 23, 24, 35, 37, 64, 72, 75, 77, 78, 79, 83, 84, 87, 88, 89, 90, 97, 98, 100,
                           101, 102, 105, 108, 110, 113, 115, 117, 118, 119, 121, 126, 127, 132, 133, 135, 136, 137, 138,
                           139, 140, 142, 143, 144, 145, 146, 148]
        drop = drop_always
        epochs.drop(drop)

    fname = directory + '/' + sub + '_comp-epo.fif'
    epochs.save(fname, overwrite=True)

def fix_comp_projs(sub, condition):
    '''
    When creating epochs from comprehension preprocessed raw file, I was using the projectors
    for R3250 for all subjects. They automatically apply when you create an evoked object
    from a raw object, so I had to remove them from the raw objects and then recreate evoked objects
    '''
    directory = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/' + sub + '/'
    raw_fname = directory + sub + '_comp_preproc.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)

    proj_fname = directory + sub + '-proj.fif'
    proj = mne.read_proj(proj_fname)

    raw.add_proj(proj, remove_existing=True)
    create_comp_epochs(raw, sub, condition, directory)

    return

def create_comp_epochs(raw, sub, condition, directory):
    comp_condition = 'X'
    prod_condition = 'X'
    if condition == 'A':
        comp_condition = 'A1'
        prod_condition = 'B1'
    elif condition == 'B':
        comp_condition = 'B1'
        prod_condition = 'A1'
    elif condition == 'C':
        comp_condition = 'B1'
        prod_condition = 'A1'
    elif condition == 'D':
        comp_condition = 'A1'
        prod_condition = 'B1'
    elif condition == 'E':
        comp_condition = 'A2'
        prod_condition = 'B2'
    elif condition == 'F':
        comp_condition = 'B2'
        prod_condition = 'A2'
    elif condition == 'G':
        comp_condition = 'B2'
        prod_condition = 'A2'
    elif condition == 'H':
        comp_condition = 'A2'
        prod_condition = 'B2'

    events = mne.find_events(raw, stim_channel="STI 014")
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        event_dict = {
            "comprehension identical": 162,
            "comprehension unrelated": 164,
        }
    else:
        event_dict = {
            "comprehension identical": 162,
            "comprehension unrelated": 164,
            "ignore": 168
        }
    epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict, preload=True)
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        if comp_condition == 'A1':
            drop_always = [1, 20, 34, 39, 41, 43, 45, 57, 62, 70, 78, 81, 82, 84, 86, 88, 91, 92, 93, 94, 95,
                           96, 97, 98, 99, 101, 102, 103, 104, 105, 109, 111, 113, 114, 115, 119, 120, 121, 122, 123,
                           125, 129, 131, 135, 136, 138, 139, 140, 142, 143]

        else:
            drop_always = [2, 14, 15, 19, 23, 24, 35, 37, 64, 72, 75, 77, 78, 79, 83, 84, 87, 88, 89, 90, 97, 98, 100,
                           101, 102, 105, 108, 110, 113, 115, 117, 118, 119, 121, 126, 127, 132, 133, 135, 136, 137,
                           138,
                           139, 140, 142, 143, 144, 145, 146, 148]
        drop = drop_always
        epochs.drop(drop)

    fname = directory + '/' + sub + '_comp-epo.fif'
    epochs.save(fname, overwrite=True)

def create_auditory_epochs(raws, sub, directory):
=======
>>>>>>> 99371e3f79805ce2c9605203223f22b6524c6a64
    '''
    Input: preprocesed raw fif file, subject number and directory for accessing files
    Output: Epoch file that contains all audio onsets in the fif file, -300ms to 1000ms, comp_aud-epo.fif and prod_aud-epo.fif

    Creates different epoch objects for production/comprehension because of different baseline activity

<<<<<<< HEAD
=======
    Will save the following files
    _prod_preproc-raw.fif: preprocessed continuous data for production block
    _comp_preproc-raw.fif: preprocessed continuous data for comprehension block
    -proj.fif: two magnetometer signal space projectors calculated from empty room data
    _prod-epo.fif: epoch object for production trials, with bad productions excluded ("production unrelated", "production identical")
    _comp-epo.fif: epoch object for comprehension trials, with no trials excluded ("comprehension unrelated", "comprehension identical")
>>>>>>> 99371e3f79805ce2c9605203223f22b6524c6a64
    '''
    raw_comp = raws[0]
    raw_prod = raws[1]
    event_dict = {
        "audio onset": 166,
        "audio offset": 167,
    }

<<<<<<< HEAD
    events = mne.find_events(raw_comp, stim_channel="STI 014")
    epochs = mne.Epochs(raw_comp, events, tmin=-0.3, tmax=1, event_id=event_dict, preload=True)

    fname = directory + '/' + sub + '_comp_aud-epo.fif'
    epochs.save(fname, overwrite=True)

    del events, epochs
=======
    print(f"Preprocessing MEG data for subject {sub} and condition {condition}...")
    comp_condition = 'X'
    prod_condition = 'X'
    if condition == 'A' or 'D':
        comp_condition = 'A1'
        prod_condition = 'B1'
    elif condition == 'B' or 'C':
        comp_condition = 'B1'
        prod_condition = 'A1'
    elif condition == 'E' or 'H':
        comp_condition = 'A2'
        prod_condition = 'B2'
    elif condition == 'F' or 'G':
        comp_condition = 'B2'
        prod_condition = 'A2'

    directory = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/' + sub + '/'

    # Empty Room Projectors
    empty_room_fname = directory + '/DAQ/' + sub + '_emptyroom.sqd'
    empty_room_raw = mne.io.read_raw_kit(empty_room_fname, preload = True)

    empty_room_raw = fix_56(empty_room_raw)

    empty_room_raw.plot()
    bads_input = input('Enter bad channels, separated by comma (MEG O11,MEG 012): ')
    bad_channels = [ch.strip() for ch in bads_input.split(',') if ch.strip()]
    empty_room_raw.info['bads'] = bad_channels
    print(f"Marked bad channels: {bad_channels}")

    empty_room_projs = mne.compute_proj_raw(empty_room_raw, n_mag = 2)
    fname = directory + sub + '-proj.fif'
    mne.write_proj(fname, empty_room_projs, overwrite = True)

    del empty_room_raw

    # PRODUCTION
    prod_file = directory + sub + '_prod-raw.fif'
    raw = mne.io.read_raw_fif(prod_file, preload=True)
    raw.plot()

    bads_input = input('Enter bad channels separated by commas (e.g., MEG 011,MEG 012): ')
    bad_channels = [ch.strip() for ch in bads_input.split(',') if ch.strip()]

    # Mark bad channels in the raw object
    raw.info['bads'] = bad_channels
    print(f"Marked bad channels: {bad_channels}")

    # Fix location of channel 56
    raw = fix_56(raw)

    # Interpolate bad channels
    raw = raw.interpolate_bads(method='MNE')  # Interpolate bads

    # Filter the ICA at a high pass of 1kHz to remove low frequency drifts,
    raw_filt = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica = ICA(n_components=30, max_iter="auto")
    ica.fit(raw_filt)
    #ica
    ica.plot_sources(raw, show_scrollbars=True)
    ica.plot_components()

    plt.show(block=True)

    ica_input = input('Enter ICA components to be removed, separated by a comma: ')
    ica_excluded = [int(ch.strip()) for ch in ica_input.split(',') if ch.strip()]

    ica.exclude = ica_excluded

    ica.apply(raw)

    del ica, raw_filt

    raw.add_proj(empty_room_projs)

    fname = directory + sub + '_prod_preproc-raw.fif'
    mne.io.Raw.save(raw, fname, overwrite=True)

    events = mne.find_events(raw, stim_channel="STI 014")
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        event_dict = {
            "production identical": 162,
            "production unrelated": 164,
        }
    else:
        event_dict = {
            "production identical": 162,
            "production unrelated": 164,
            "ignore":168
        }
    epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict, preload=True)
    excel_fname = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/Behavioral_Data/Productions_' + prod_condition + '.xlsx'
    df = pd.read_excel(excel_fname)
    column_name = sub + ' Accuracy'
    bad_productions = df.index[df[column_name] == False].tolist()
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        if prod_condition == 'A1':
            drop_always = [1, 20, 34, 39, 41, 43, 45, 57, 62, 70, 78, 81, 82, 84, 86, 88, 91, 92, 93, 94, 95,
                           96, 97, 98, 99, 101, 102, 103, 104, 105, 109, 111, 113, 114, 115, 119, 120, 121, 122, 123,
                           125, 129, 131, 135, 136, 138, 139, 140, 142, 143]
        else:
            drop_always = [2, 14, 15, 19, 23, 24, 35, 37, 64, 72, 75, 77, 78, 79, 83, 84, 87, 88, 89, 90, 97, 98, 100,
                           101,102, 105, 108, 110, 113, 115, 117, 118, 119, 121, 126, 127, 132, 133, 135, 136, 137, 138,
                           139, 140, 142, 143, 144, 145, 146, 148]
        drop = np.unique(np.concatenate((bad_productions, drop_always)))
    else:
        drop = bad_productions
    epochs.drop(drop)

    fname = directory + '/' + sub + '_prod-epo.fif'
    epochs.save(fname, overwrite=True)

    # del raw, epochs, raw_filt, ica
>>>>>>> 99371e3f79805ce2c9605203223f22b6524c6a64

    events= mne.find_events(raw_prod, stim_channel="STI 014")
    epochs = mne.Epochs(raw_prod, events, tmin=-0.3, tmax=1, event_id=event_dict, preload=True)

<<<<<<< HEAD
    fname = directory + '/' + sub + '_prod_aud-epo.fif'
    epochs.save(fname, overwrite=True)

    return 0
=======
    #COMPREHENSION
    comp_file = directory + sub + '_comp-raw.fif'
    raw = mne.io.read_raw_fif(comp_file, preload=True)
    raw.plot()

    bads_input = input('Enter bad channels separated by commas (e.g., MEG 011,MEG 012): ')
    bad_channels = [ch.strip() for ch in bads_input.split(',') if ch.strip()]

    # Mark bad channels in the raw object
    raw.info['bads'] = bad_channels
    print(f"Marked bad channels: {bad_channels}")

    # Fix the location of MEG 056 (always)
    loc = np.array([
        0.09603, -0.07437, 0.00905, -0.5447052, -0.83848277,
        0.01558496, 0., -0.01858388, -0.9998273, 0.8386276,
        -0.54461113, 0.01012274])
    index = raw.ch_names.index('MEG 056')
    raw.info['chs'][index]['loc'] = loc

    # Interpolate bad channels
    raw = raw.interpolate_bads(method='MNE')  # Interpolate bads

    # Filter the ICA at a high pass of 1kHz to remove low frequency drifts,
    # We will apply the ICA to the unfiltered data, though, so we keep a copy
    raw_filt = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica = ICA(n_components=30, max_iter="auto")
    ica.fit(raw_filt)

    ica.plot_sources(raw, show_scrollbars=True)
    ica.plot_components()

    plt.show(block=True)

    ica_input = input('Enter ICA components to be removed, separated by a comma: ')
    ica_excluded = [int(ch.strip()) for ch in ica_input.split(',') if ch.strip()]

    ica.exclude = ica_excluded

    ica.apply(raw)

    del ica, raw_filt

    raw.add_proj(empty_room_projs)

    fname = directory + '/' + sub + '_comp_preproc-raw.fif'
    mne.io.Raw.save(raw, fname, overwrite=True)

    events = mne.find_events(raw, stim_channel="STI 014")
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        event_dict = {
            "comprehension identical": 162,
            "comprehension unrelated": 164,
        }
    else:
        event_dict = {
            "comprehension identical": 162,
            "comprehension unrelated": 164,
            "ignore": 168
        }
    epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict, preload=True)
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        if comp_condition=='A1':
            drop_always = [1, 20, 34, 39, 41, 43, 45, 57, 62, 70, 78, 81, 82, 84, 86, 88, 91, 92, 93, 94, 95,
                           96, 97, 98, 99, 101, 102, 103, 104, 105, 109, 111, 113, 114, 115, 119, 120, 121, 122, 123,
                           125, 129, 131, 135, 136, 138, 139, 140, 142, 143]

        else:
            drop_always = [2, 14, 15, 19, 23, 24, 35, 37, 64, 72, 75, 77, 78, 79, 83, 84, 87, 88, 89, 90, 97, 98, 100,
                           101, 102, 105, 108, 110, 113, 115, 117, 118, 119, 121, 126, 127, 132, 133, 135, 136, 137, 138,
                           139, 140, 142, 143, 144, 145, 146, 148]
        drop = drop_always
        epochs.drop(drop)

    fname = directory + '/' + sub + '_comp-epo.fif'
    epochs.save(fname, overwrite=True)

def preprocess_meg_maxwell(sub, condition):
    comp_condition = 'X'
    prod_condition = 'X'
    if condition in ['A', 'D']:
        comp_condition = 'A1'
        prod_condition = 'B1'
    elif condition in ['B','C']:
        comp_condition = 'B1'
        prod_condition = 'A1'
    elif condition in ['E','H']:
        comp_condition = 'A2'
        prod_condition = 'B2'
    elif condition in ['F', 'G']:
        comp_condition = 'B2'
        prod_condition = 'A2'

    print(prod_condition)
    input_dir = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/' + sub + '/'
    output_dir = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/Testing/' + sub + '/'

    bads = ['MEG 056', 'MEG 086']

    comp_raw_fname = input_dir + sub + '_comp-raw.fif'
    prod_raw_fname = input_dir + sub + '_prod-raw.fif'
    empty_room_fname = input_dir + '/DAQ/' + sub + '_emptyroom.sqd'

    comp_raw = mne.io.read_raw_fif(comp_raw_fname, preload=True)
    prod_raw = mne.io.read_raw_fif(prod_raw_fname, preload=True)
    empty_room_raw = mne.io.read_raw_kit(empty_room_fname, preload=True)
    empty_room_raw = mne.io.RawArray(empty_room_raw.get_data(), comp_raw.info)

    comp_raw = fix_56(comp_raw)
    prod_raw = fix_56(prod_raw)
    empty_room_raw = fix_56(empty_room_raw)

    noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(
        comp_raw, ignore_ref=True
    )
    for i in bads:
        if i not in noisy_chs:
            noisy_chs.append(i)

    comp_raw.info['bads'] = noisy_chs
    prod_raw.info['bads'] = noisy_chs
    empty_room_raw.info['bads'] = noisy_chs

    comp_raw_tsss = mne.preprocessing.maxwell_filter(
        comp_raw,
        st_duration=10,
        ignore_ref=True,
        st_correlation=0.9,
        st_only=True
    )

    prod_raw_tsss = mne.preprocessing.maxwell_filter(
        prod_raw,
        st_duration=10,
        ignore_ref=True,
        st_correlation=0.9,
        st_only=True
    )

    empty_room_raw_tsss = mne.preprocessing.maxwell_filter(
        empty_room_raw,
        st_duration=10,
        ignore_ref=True,
        st_correlation=0.9,
        st_only=True
    )
    os.makedirs(output_dir, exist_ok=True)
    fname = output_dir + '/' + sub + '_emptyroom_preproc-raw.fif'
    mne.io.Raw.save(empty_room_raw_tsss, fname, overwrite=True)


    #ICA FOR COMPREHENSION
    comp_raw_tsss_filt = comp_raw_tsss.copy().filter(l_freq=1.0, h_freq=None)
    ica = ICA(n_components=30, max_iter="auto")
    ica.fit(comp_raw_tsss_filt)

    ica.plot_sources(comp_raw_tsss, show_scrollbars=True)
    ica.plot_components()

    ica_input = input('Enter ICA components to be removed, separated by a comma: ')
    ica_excluded = [int(ch.strip()) for ch in ica_input.split(',') if ch.strip()]

    ica.exclude = ica_excluded
    ica.apply(comp_raw_tsss)

    fname = output_dir + '/' + sub + '_comp_preproc-raw.fif'
    mne.io.Raw.save(comp_raw_tsss, fname, overwrite=True)

    #ICA FOR PRODUCTION
    prod_raw_tsss_filt = prod_raw_tsss.copy().filter(l_freq=1.0, h_freq=None)
    ica = ICA(n_components=30, max_iter="auto")
    ica.fit(prod_raw_tsss_filt)

    ica.plot_sources(prod_raw_tsss, show_scrollbars=True)
    ica.plot_components()

    ica_input = input('Enter ICA components to be removed, separated by a comma: ')
    ica_excluded = [int(ch.strip()) for ch in ica_input.split(',') if ch.strip()]

    ica.exclude = ica_excluded
    ica.apply(prod_raw_tsss)

    fname = output_dir + '/' + sub + '_prod_preproc-raw.fif'
    mne.io.Raw.save(prod_raw_tsss, fname, overwrite=True)

    #EPOCHING COMPREHENSION
    events = mne.find_events(comp_raw_tsss, stim_channel="STI 014")
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        event_dict = {
            "comprehension identical": 162,
            "comprehension unrelated": 164,
        }
    else:
        event_dict = {
            "comprehension identical": 162,
            "comprehension unrelated": 164,
            "ignore": 168
        }
    epochs = mne.Epochs(comp_raw_tsss, events, tmin=-0.3, tmax=0.7, event_id=event_dict, preload=True)
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        if comp_condition == 'A1':
            drop_always = [1, 20, 34, 39, 41, 43, 45, 57, 62, 70, 78, 81, 82, 84, 86, 88, 91, 92, 93, 94, 95,
                           96, 97, 98, 99, 101, 102, 103, 104, 105, 109, 111, 113, 114, 115, 119, 120, 121, 122, 123,
                           125, 129, 131, 135, 136, 138, 139, 140, 142, 143]

        else:
            drop_always = [2, 14, 15, 19, 23, 24, 35, 37, 64, 72, 75, 77, 78, 79, 83, 84, 87, 88, 89, 90, 97, 98, 100,
                           101, 102, 105, 108, 110, 113, 115, 117, 118, 119, 121, 126, 127, 132, 133, 135, 136, 137,
                           138,
                           139, 140, 142, 143, 144, 145, 146, 148]
        drop = drop_always
        epochs.drop(drop)

    fname = output_dir + '/' + sub + '_comp-epo.fif'
    epochs.save(fname, overwrite=True)

    #EPOCHING PRODUCTION
    events = mne.find_events(prod_raw_tsss, stim_channel="STI 014")
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        event_dict = {
            "production identical": 162,
            "production unrelated": 164,
        }
    else:
        event_dict = {
            "production identical": 162,
            "production unrelated": 164,
            "ignore": 168
        }
    epochs = mne.Epochs(prod_raw_tsss, events, tmin=-0.3, tmax=0.7, event_id=event_dict, preload=True)
    excel_fname = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/Behavioral_Data/Productions_' + prod_condition + '.xlsx'
    df = pd.read_excel(excel_fname)
    column_name = sub + ' Accuracy'
    bad_productions = df.index[df[column_name] == False].tolist()
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        if prod_condition == 'A1':
            drop_always = [1, 20, 34, 39, 41, 43, 45, 57, 62, 70, 78, 81, 82, 84, 86, 88, 91, 92, 93, 94, 95,
                           96, 97, 98, 99, 101, 102, 103, 104, 105, 109, 111, 113, 114, 115, 119, 120, 121, 122, 123,
                           125, 129, 131, 135, 136, 138, 139, 140, 142, 143]
        else:
            drop_always = [2, 14, 15, 19, 23, 24, 35, 37, 64, 72, 75, 77, 78, 79, 83, 84, 87, 88, 89, 90, 97, 98, 100,
                           101, 102, 105, 108, 110, 113, 115, 117, 118, 119, 121, 126, 127, 132, 133, 135, 136, 137,
                           138,
                           139, 140, 142, 143, 144, 145, 146, 148]
        drop = np.unique(np.concatenate((bad_productions, drop_always)))
    else:
        drop = bad_productions
    epochs.drop(drop)

    fname = output_dir + '/' + sub + '_prod-epo.fif'
    epochs.save(fname, overwrite=True)


def fix_comp_projs(sub, condition):
    '''
    When creating epochs from comprehension preprocessed raw file, I was using the projectors
    for R3250 for all subjects. They automatically apply when you create an evoked object
    from a raw object, so I had to remove them from the raw objects and then recreate evoked objects
    '''
    directory = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/' + sub + '/'
    raw_fname = directory + sub + '_comp_preproc.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
>>>>>>> 99371e3f79805ce2c9605203223f22b6524c6a64

def main():
    for sub in ['R3250', 'R3254', 'R3261', 'R3264', 'R3270','R3271','R3272','R3273','R3275','R3277','R3279','R3285','R3286','R3289','R3290']:
        directory = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/' + sub + '/'

        comp_fname = directory + '/' + sub + '_comp_preproc-raw.fif'
        prod_fname = directory + '/' + sub + '_prod_preproc-raw.fif'
        comp_epochs = mne.io.read_raw_fif(comp_fname)
        prod_epochs = mne.io.read_raw_fif(prod_fname)

<<<<<<< HEAD
        create_auditory_epochs([comp_epochs, prod_epochs], sub, directory)

    return 0

preprocess_meg('R3250', 'B')
=======
    return

def create_comp_epochs(raw, sub, condition, directory):
    comp_condition = 'X'
    if condition == 'A' or 'D':
        comp_condition = 'A1'
    elif condition == 'B' or 'C':
        comp_condition = 'B1'
    elif condition == 'E' or 'H':
        comp_condition = 'A2'
    elif condition == 'F' or 'G':
        comp_condition = 'B2'

    events = mne.find_events(raw, stim_channel="STI 014")
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        event_dict = {
            "comprehension identical": 162,
            "comprehension unrelated": 164,
        }
    else:
        event_dict = {
            "comprehension identical": 162,
            "comprehension unrelated": 164,
            "ignore": 168
        }
    epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict, preload=True)
    if sub in ['R3250', 'R3254', 'R3260', 'R3261', 'R3264']:
        if comp_condition == 'A1':
            drop_always = [1, 20, 34, 39, 41, 43, 45, 57, 62, 70, 78, 81, 82, 84, 86, 88, 91, 92, 93, 94, 95,
                           96, 97, 98, 99, 101, 102, 103, 104, 105, 109, 111, 113, 114, 115, 119, 120, 121, 122, 123,
                           125, 129, 131, 135, 136, 138, 139, 140, 142, 143]

        else:
            drop_always = [2, 14, 15, 19, 23, 24, 35, 37, 64, 72, 75, 77, 78, 79, 83, 84, 87, 88, 89, 90, 97, 98, 100,
                           101, 102, 105, 108, 110, 113, 115, 117, 118, 119, 121, 126, 127, 132, 133, 135, 136, 137,
                           138,
                           139, 140, 142, 143, 144, 145, 146, 148]
        drop = drop_always
        epochs.drop(drop)

    fname = directory + '/' + sub + '_comp-epo.fif'
    epochs.save(fname, overwrite=True)

def create_auditory_epochs(raws, sub, directory):
    '''
    Input: preprocesed raw fif file, subject number and directory for accessing files
    Output: Epoch file that contains all audio onsets in the fif file, -300ms to 1000ms, comp_aud-epo.fif and prod_aud-epo.fif

    Creates different epoch objects for production/comprehension because of different baseline activity

    '''
    raw_comp = raws[0]
    raw_prod = raws[1]
    event_dict = {
        "audio onset": 166,
        "audio offset": 167,
    }

    events = mne.find_events(raw_comp, stim_channel="STI 014")
    epochs = mne.Epochs(raw_comp, events, tmin=-0.3, tmax=1, event_id=event_dict, preload=True)

    fname = directory + '/' + sub + '_comp_aud-epo.fif'
    epochs.save(fname, overwrite=True)

    del events, epochs

    events= mne.find_events(raw_prod, stim_channel="STI 014")
    epochs = mne.Epochs(raw_prod, events, tmin=-0.3, tmax=1, event_id=event_dict, preload=True)

    fname = directory + '/' + sub + '_prod_aud-epo.fif'
    epochs.save(fname, overwrite=True)

    return 0

def main():
    for sub in ['R3250', 'R3254', 'R3261', 'R3264', 'R3270','R3271','R3272','R3273','R3275','R3277','R3279','R3285','R3286','R3289','R3290','R3285','R3286','R3289','R3290']:
        directory = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/' + sub + '/'

        comp_fname = directory + '/' + sub + '_comp_preproc-raw.fif'
        prod_fname = directory + '/' + sub + '_prod_preproc-raw.fif'
        comp_epochs = mne.io.read_raw_fif(comp_fname)
        prod_epochs = mne.io.read_raw_fif(prod_fname)

        create_auditory_epochs([comp_epochs, prod_epochs], sub, directory)

    return 0

preprocess_meg_maxwell('R3290', 'H')
>>>>>>> 99371e3f79805ce2c9605203223f22b6524c6a64

