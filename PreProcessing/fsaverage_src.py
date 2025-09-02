import mne
from os import path as op

# Define your SUBJECTS_DIR
subjects_dir = '/Applications/freesurfer/8.0.0/subjects'

# Set the spacing to 'ico4' (icosahedral subdivision grade 4)
spacing = 'ico4'

# Create the source space on the 'fsaverage' subject
src_ref = mne.setup_source_space(
    subject='fsaverage',
    spacing=spacing,
    subjects_dir=subjects_dir,
    add_dist=False # Set to True for more details or if needed later
)

# Define the output filename
fname_src = op.join(subjects_dir, 'fsaverage','bem', 'fsaverage-ico-4-src.fif')

# Save the source space
mne.write_source_spaces(fname_src, src_ref, overwrite=True)

print(f"The source space file has been saved to: {fname_src}")