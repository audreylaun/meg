import mne
import matplotlib.pyplot as plt

# left posterior
left_post_numbers = [4, 5, 6, 7, 8, 9, 34, 36, 37, 38, 40, 47, 48, 49, 50, 75, 76, 77, 79, 88, 127, 129,
                     137, 89, 92, 94,12, 10, 11,35,46,51,72,73,74,78,91,125,126,138,140,141,128,41]
left_post = []
for i in left_post_numbers:
    title = ""
    if i < 10:
        title = 'MEG 00' + str(i)
    elif 10 <= i < 100:
        title = 'MEG 0' + str(i)
    else:
        title = 'MEG ' + str(i)
    left_post.append(title)

#left anterior
left_ant_numbers = [1, 2, 3, 39, 42, 43, 44, 80, 81,86,83, 84, 85, 108, 130, 131, 132, 133, 134, 135,
                    136, 151, 65, 59, 152, 53, 68, 143,105,106,107,109,45,111]
left_ant = []
for i in left_ant_numbers:
    title = ""
    if i < 10:
        title = 'MEG 00' + str(i)
    elif 10 <= i < 100:
        title = 'MEG 0' + str(i)
    else:
        title = 'MEG ' + str(i)
    left_ant.append(title)
print(left_ant)

#right posterior
right_post_numbers = [14, 15, 16, 17, 18, 19, 27, 28, 30, 54, 56, 57, 66, 69, 70, 97, 119, 121, 122,
                      90, 87, 71, 52, 82, 58, 67, 95, 26, 145, 13,29,31,32,33,120,123,124,142]
right_post = []
for i in right_post_numbers:
    title = ""
    if i < 10:
        title = 'MEG 00' + str(i)
    elif 10 <= i < 100:
        title = 'MEG 0' + str(i)
    else:
        title = 'MEG ' + str(i)
    right_post.append(title)

#right anterior
right_ant_numbers = [20, 21, 22, 23, 24, 60, 61, 63, 99, 100, 114, 115, 116, 117, 118, 147,
                     148, 155, 96, 25,62,64,98,101,102,103,104,112,113,119,93]
right_ant = []
for i in right_ant_numbers:
    title = ""
    if i < 10:
        title = 'MEG 00' + str(i)
    elif 10 <= i < 100:
        title = 'MEG 0' + str(i)
    else:
        title = 'MEG ' + str(i)
    right_ant.append(title)

# plotting sensor locations
raw = mne.io.read_raw_fif('/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/R3250/R3250_comp-raw.fif', preload=True)
raw_post = raw.pick(right_post)
fig = plt.figure()
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection="3d")
raw_post.plot_sensors(ch_type="mag", axes=ax2d)
raw_post.plot_sensors(ch_type="mag", axes=ax3d, kind="3d")
ax3d.view_init(azim=70, elev=15)
fig.show()

raw = mne.io.read_raw_fif('/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/R3250/R3250_comp-raw.fif', preload=True)
raw_post = raw.pick(left_post)
fig = plt.figure()
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection="3d")
raw_post.plot_sensors(ch_type="mag", axes=ax2d)
raw_post.plot_sensors(ch_type="mag", axes=ax3d, kind="3d")
ax3d.view_init(azim=70, elev=15)
fig.show()

raw = mne.io.read_raw_fif('/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/R3250/R3250_comp-raw.fif', preload=True)
raw_post = raw.pick(left_ant)
fig = plt.figure()
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection="3d")
raw_post.plot_sensors(ch_type="mag", axes=ax2d)
raw_post.plot_sensors(ch_type="mag", axes=ax3d, kind="3d")
ax3d.view_init(azim=70, elev=15)
fig.show()

raw = mne.io.read_raw_fif('/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/R3250/R3250_comp-raw.fif', preload=True)
raw_ant = raw.pick(right_ant)
fig = plt.figure()
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection="3d")
raw_ant.plot_sensors(ch_type="mag", axes=ax2d)
raw_ant.plot_sensors(ch_type="mag", axes=ax3d, kind="3d")
ax3d.view_init(azim=70, elev=15)
fig.show()

plt.show()