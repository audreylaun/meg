import mne
import numpy as np
from matplotlib import pyplot as plt
from mne.preprocessing import ICA, corrmap
import argparse
import pandas as pd

def fix_56(raw):
    # Fix the location of MEG 056
    loc = np.array([
        0.09603, -0.07437, 0.00905, -0.5447052, -0.83848277,
        0.01558496, 0., -0.01858388, -0.9998273, 0.8386276,
        -0.54461113, 0.01012274])
    index = raw.ch_names.index('MEG 056')
    raw.info['chs'][index]['loc'] = loc
    return raw

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

    fname = directory + '/' + sub + '_comp_TEST-epo.fif'
    epochs.save(fname, overwrite=True)


def preprocess_meg(sub, condition):
    '''
    sub = subject number
    condition = condition number (A,B,C,D,E,F,G,H)

    Will access the following files:
    _prod-raw.fif, _comp-raw.fif, emptyroom.sqd

    Will save the following files
    _prod_preproc.fif: preprocessed continuous data for production block
    _comp_preproc.fif: preprocessed continuous data for comprehension block
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

    directory = '/Users/admin/Box Sync/Starling/Experiment1/MEG_data/' + sub + '/'

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

    fig = plt.figure()
    ax2d = fig.add_subplot(121)
    ax3d = fig.add_subplot(122, projection="3d")
    raw.plot_sensors(ch_type="mag", axes=ax2d)
    raw.plot_sensors(ch_type="mag", axes=ax3d, kind="3d")
    ax3d.view_init(azim=70, elev=15)

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

    fname = directory + sub + '_prod_preproc.fif'
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
    excel_fname = '/Users/admin/Box Sync/Starling/Experiment1/Behavioral_Data/Productions_' + prod_condition + '.xlsx'
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

    fname = directory + '/' + sub + '_prod_TEST-epo.fif'
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

    # empty_room_projs = mne.read_proj('/Users/admin/Box Sync/Starling/Experiment1/MEG_data/R3250/R3250-proj.fif')
    raw.add_proj(empty_room_projs)

    fname = directory + '/' + sub + '_comp_preproc.fif'
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

    fname = directory + '/' + sub + '_comp_TEST-epo.fif'
    epochs.save(fname, overwrite=True)

def fix_comp_projs(sub, condition):
    '''
    When creating epochs from comprehension preprocessed raw file, I was using the projectors
    for R3250 for all subjects. They automatically apply when you create an evoked object
    from a raw object, so I had to remove them from the raw objects and then recreate evoked objects
    '''
    directory = '/Users/admin/Box Sync/Starling/Experiment1/MEG_data/' + sub + '/'
    raw_fname = directory + sub + '_comp_preproc.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)

    proj_fname = directory + sub + '-proj.fif'
    proj = mne.read_proj(proj_fname)

    raw.add_proj(proj, remove_existing=True)
    create_comp_epochs(raw, sub, condition, directory)

    return



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Preprocess MEG data for a subject.")
    # parser.add_argument("sub", type=str, help="Subject number (e.g., 101)")
    # parser.add_argument("condition", type=str, help="Condition (A1, B1, A2, B2")
    #
    #
    # args = parser.parse_args()
    # preprocess_meg(args.sub, args.condition)
    subs = ['R3250', 'R3254', 'R3261', 'R3264', 'R3270','R3271','R3272','R3273','R3275','R3277','R3279','R3285','R3286','R3289','R3290']
    conditions = ['B', 'C', 'D','A','D','A','B','C','D','B','C','E','F','G','H']
    # for i in range(len(subs)):
    #     sub = subs[i]
    #     directory = '/Users/admin/Box Sync/Starling/Experiment1/MEG_data/' + sub + '/'
    #     fname = directory + '/' + sub + '_comp_TEST-epo.fif'
    #     epoch = mne.read_epochs(fname, verbose = False)
    #     print(epoch)
    for i in range(len(subs)):
        sub = subs[i]
        directory = '/Users/admin/Box Sync/Starling/Experiment1/MEG_data/' + sub + '/'

        print(sub)
        condition = conditions[i]
        fix_comp_projs(sub, condition)
        fname = directory + '/' + sub + '_comp_TEST-epo.fif'
        epoch = mne.read_epochs(fname, verbose=False)
        print(epoch)



