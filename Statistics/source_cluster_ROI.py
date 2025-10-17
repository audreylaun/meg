# Spatio-temporal permutation test on MEG data (in source space).

# FOLDERS STRUCTURE:
# > OUT
#   > Results
#       > ObjectNaming
#       > WordReading


#====================Import data into eelbrain.Dataset=========================#
ROOT = f'/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/Testing/'
OUT = f'/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/MEG_data/Testing/'

region = 'temporal'
# region = 'frontal'

# hemi = 'lh'
hemi = 'rh'

import mne, os, eelbrain, pickle
from eelbrain import plot
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

subjects_dir = '/Applications/freesurfer/8.0.0/subjects'
subjects =['R3250', 'R3254', 'R3261', 'R3264', 'R3270','R3271','R3272','R3273','R3275','R3277','R3279','R3285','R3286','R3289','R3290','R3285','R3286','R3289','R3290','R3326','R3327','R3329']

#Create/load fsaverage source space:
if not os.path.isfile('/Applications/freesurfer/8.0.0/subjects/fsaverage/bem/fsaverage-ico-4-src.fif'):
    src = mne.setup_source_space(subject='fsaverage', spacing='ico4', subjects_dir=subjects_dir)
    del src

src_fname = '/Applications/freesurfer/8.0.0/subjects/fsaverage/bem/fsaverage-ico-4-src.fif'
src = mne.read_source_spaces(src_fname)

stcs, prime_type, task, subject = [],[],[],[]
#---------------------------load stcs of all subjs-----------------------------#
for subj in subjects:
    stc_path = ROOT+subj+'/'
    labels = [i for i in os.listdir(stc_path) if i.endswith(f"-{hemi}.stc")]
    for i in labels:
        stc = mne.read_source_estimate(stc_path + i)
        morph = mne.compute_source_morph(
            stc,
            subject_from=subj,
            subject_to='fsaverage',
            src_to=src,
            subjects_dir=subjects_dir)
        stc = morph.apply(stc)
        print(stc.times, stc.vertices)
        print(len(stc.times), len(stc.vertices))
        stcs.append(stc)
        prime_type.append(str.split(i,'_')[1])
        task.append(str.split(i,'_')[2][:4])
        subject.append(subj)
        del stc

#--------------------Loading into an eelbrain.Dataset--------------------------#
ds = eelbrain.Dataset()
ds['stc'] = eelbrain.load.fiff.stc_ndvar(stcs,subject='fsaverage',src='ico-4',subjects_dir=subjects_dir,method='dSPM',fixed=False,parc='aparc.a2009s')
ds['Task'] = eelbrain.Factor(task)
ds['Prime_Type'] = eelbrain.Factor(prime_type)
ds['subject']=eelbrain.Factor(subject,random=True)

src=ds['stc']

#==============================================================================#
#                           Done importing data                                #
#==============================================================================#

#-----------------------cluster based permutation test-------------------------#
# condition names
Tasks = ['prod', 'comp']
Prime_Types = ['ident', 'unrel']
pvalue = 0.15

src = ds['stc']

os.environ["SUBJECTS_DIR"] = '/Applications/freesurfer/8.0.0/subjects'

for current_task in Tasks:
    other_task = [t for t in Tasks if t is not current_task][0]
    ds['stc']=src #reset data to full space
    src = eelbrain.set_parc(src, 'aparc.a2009s')

    if region == "frontal":
        frontal_labels_old = ["Frontal",
                              [f"G_and_S_frontomargin-{hemi}", f"G_and_S_precentral-{hemi}", f"G_front_middle-{hemi}",
                               f"G_front_sup-{hemi}", f"G_opercularis-{hemi}", f"G_orbital-{hemi}",
                               f"G_subcentral-{hemi}",
                               f"G_suborbital-{hemi}", f"Pole_frontal-{hemi}", f"S_front_inf-{hemi}",
                               f"S_front_middle-{hemi}",
                               f"S_orbital_lateral-{hemi}", f"S_orbital_med-olfact-{hemi}",
                               f"S_orbital-H_Shaped-{hemi}",
                               f"S_precentral-inf-part-{hemi}", f"S_precentral-sup-part-{hemi}",
                               f"S_subcentral_ant-{hemi}",
                               f"S_subcentral_post-{hemi}", f"S_front_sup-{hemi}"]]

        frontal_labels_1 = mne.read_labels_from_annot('fsaverage', 'aparc.a2009s', hemi, 'white', None, 'front',
                                                      '/Applications/freesurfer/8.0.0/subjects')
        frontal_labels_2 = mne.read_labels_from_annot('fsaverage', 'aparc.a2009s', hemi, 'white', None, 'precentral',
                                                      '/Applications/freesurfer/8.0.0/subjects')
        frontal_labels_3 = mne.read_labels_from_annot('fsaverage', 'aparc.a2009s', hemi, 'white', None, 'S_central',
                                                      '/Applications/freesurfer/8.0.0/subjects')

        # Get all the labels that were in the original frontal plotting names
        all_labels = mne.read_labels_from_annot('fsaverage', 'aparc.a2009s', hemi, 'white', None, None,
                                                '/Applications/freesurfer/8.0.0/subjects')
        frontal_labels_plot = []
        for i in all_labels:
            if i.name in frontal_labels_old[1]:
                frontal_labels_plot.append(i)

        # Get all the labels that the original frontal plotting was missing
        labels_temp = []
        for i in frontal_labels_1:
            labels_temp.append(i)
        for i in frontal_labels_2:
            labels_temp.append(i)
        for i in frontal_labels_3:
            labels_temp.append(i)

        for i in labels_temp:
            if i.name not in [lab.name for lab in frontal_labels_plot]:
                frontal_labels_plot.append(i)

        label = sum(frontal_labels_plot[1:], frontal_labels_plot[0])

    elif region == "temporal":
        labels_temp = mne.read_labels_from_annot('fsaverage', 'aparc.a2009s', hemi, 'white', None, 'temp',
                                                 '/Applications/freesurfer/8.0.0/subjects')
        label = sum(labels_temp[1:], labels_temp[0])

    src_region = src.sub(source=label) #reducing the ds to just the sources of interest. can also sub with time.
    ds['stc']=src_region

    res = eelbrain.testnd.TTestRelated('stc', x='Prime_Type', match='subject',sub=ds['Task']==current_task,ds=ds,tstart=0,tstop=0.6,tail=1, samples=10000,pmin=.05,mintime=0.01)
    print(res.clusters)
    pickle.dump(res, open(OUT+'Results/%s/res.p'%current_task,'wb'))
    f=open(OUT + 'Results/%s/%s_results_table_%s.txt' %(current_task,hemi,region), 'w')
    f.write('Model: %s, N=%s\n' %(res.x, len(subjects)))
    f.write('tstart=%s, tstop=%s, samples=%s, pmin=%s\n\n' %(res.tstart, res.tstop, res.samples, res.pmin))
    f.write(str(res.clusters))
    f.close()

    ix_sign_clusters=np.where(res.clusters['p']<=pvalue)[0]

    for i in range(len(ix_sign_clusters)):
        cluster = res.clusters[ix_sign_clusters[i]]['cluster']
        tstart = res.clusters[ix_sign_clusters[i]]['tstart']
        tstop = res.clusters[ix_sign_clusters[i]]['tstop']

        #save significant cluster as a label for plotting.
        label = eelbrain.labels_from_clusters(cluster)
        label[0].name = f"label-{hemi}"
        mne.write_labels_to_annot(label,subject='fsaverage', parc='cluster%s_FullAnalysis'%i ,subjects_dir=subjects_dir, overwrite=True)
        src = eelbrain.set_parc(src, 'cluster%s_FullAnalysis' %i)
        src_region = src.sub(source=label[0].name)
        ds['stc']=src_region
        timecourse = src_region.mean('source')

        # Set colors for first and second plots
        if current_task == "prod":
            shade_color = (1, 0.75, 0.75)
            colors1 = {
                ('ident'): plot.Style((1, 0, 0)),
                ('unrel'): plot.Style((1,  0, 0), linestyle='--'),
            }
            colors2 = {
                ('ident'): plot.Style((0, 0, 1)),
                ('unrel'): plot.Style((0, 0, 1), linestyle='--'),
            }
        else:
            shade_color = (0.75, 0.75, 1)
            colors1 = {
                ('ident'): plot.Style((0, 0, 1)),
                ('unrel'): plot.Style((0, 0, 1), linestyle='--'),
            }
            colors2 = {
                ('ident'): plot.Style((1, 0, 0)),
                ('unrel'): plot.Style((1,  0, 0), linestyle='--'),
            }
        colors_all = {
            ('ident', 'prod'): plot.Style((1, 0, 0)),
            ('unrel', 'prod'): plot.Style((1, 0, 0), linestyle='--'),
            ('ident', 'comp'): plot.Style((0, 0, 1)),
            ('unrel', 'comp'): plot.Style((0, 0, 1), linestyle='--'),
        }


        # Plot
            # 1)Timecourse
        activation = eelbrain.plot.UTSStat(timecourse,x='Prime_Type', ds=ds, sub=ds['Task']==current_task, colors=colors1, legend='lower left')
        activation.add_vspan(xmin=tstart, xmax=tstop, color='lightgrey', zorder=-50)
        activation.save(OUT+'Results/%s/%s_cluster%s_timecourse_%s_(%s-%s).png' %(current_task, hemi, i+1,region, tstart, tstop))
        activation.close()


        activation = eelbrain.plot.UTSStat(timecourse, x='Prime_Type % Task', ds=ds,colors=colors_all, legend='lower left')
        activation.add_vspan(xmin=tstart, xmax=tstop, color=shade_color, zorder=-50)
        activation.save(OUT+'Results/%s/%s_cluster%s_timecourse_all_%s_(%s-%s).png' %(current_task, hemi, i+1,region, tstart, tstop))
        activation.close()

            # 2) Brain
        brain = eelbrain.plot.brain.cluster(cluster.mean('time'))
        brain.save_image(OUT+'Results/%s/%s_cluster%s_brain_%s_(%s-%s)_Task=%s.png' %(current_task, hemi, i+1, region, tstart, tstop, current_task))
        brain.close()

        #     # 3) Bar graph
        # ds['average_source_activation'] = timecourse.mean(time=(tstart,tstop)) #!!! should be restrained to my sign timewindow
        # bar = eelbrain.plot.Barplot(ds['average_source_activation'], X=effect, ds=ds, sub=ds['Task']==current_task)
        # bar.save(OUT+'Results/%s/cluster%s_BarGraph_(%s-%s)_Task=%s,effect=%s.png'%(current_task,i+1, tstart, tstop, current_task, effect))
        # bar.close()



#==============================================================================#
#==============================================================================#