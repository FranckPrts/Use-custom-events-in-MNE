

#%%
import hypyp

import mne
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
import seaborn as sns

data_path    = "../../W2W-Data/W2W-EEG/W2W-Brittany-Data/"
save_path    = "../../W2W-save/"

#%%
# run this for interactive plots
%matplotlib qt
# run this for static plots
# %matplotlib inline 

#%%
all_file = os.listdir(data_path) 
try:
    all_file.remove('.DS_Store')
except:
    pass

#%%
# Get unique dyad number for a given context
context = 'movie' # form book puzzle movie

dyad_nb=[]
for i in all_file:
    if i.split('_')[2] == context:
        dyad_nb.append(i.split('_')[1])
    else:
        pass
dyad_nb = np.unique(dyad_nb)

#%%

# Also, for the analysis metrics, first priority is PLV. 
# Let's do 8-9 hz in P3, Pz, P4, C3, C4, P03, P04 - can we 
# keep electrodes separate for now? If that's too challenging, 
# we can prioritize P3 and P4. 

# Second priority would be Power Correlation in those same electrodes. 
# Hypyp documentation https://hypyp.readthedocs.io/en/latest/


ibc_metric = 'plv'
selected_chans = ['P3', 'Pz', 'P4', 'C3', 'C4', 'P03', 'P04']
n_ch = len(selected_chans)
freq_bands = {'Theta': [8.0, 9.0]}
nb_freq_band = len(freq_bands.keys())
nb_of_dyad = len(dyad_nb)
#%%
# Create the dict where the IBC will be stored

# results_IBC = np.zeros([nb_of_dyad, nb_freq_band, n_ch*2, n_ch*2], dtype=np.float32)
results_IBC = {}

#%%
non_epoch_dyads  = []
no_dropChan_dyad = []


for dyad in dyad_nb:
    
    print(dyad)
    
    # load the data
    print('{}W2W_{}_{}_hpfilt_chlocs_childinterp_parentinterp_binned_runICA_componentsremoved_childAR_parentAR_selectedepochs.set'.format(
        data_path, dyad, context))
    try:
        epo = mne.read_epochs_eeglab('{}W2W_{}_{}_hpfilt_chlocs_childinterp_parentinterp_binned_runICA_componentsremoved_childAR_parentAR_selectedepochs.set'.format(
        data_path, dyad, context))
    except:
        non_epoch_dyads.append(dyad)

    try:
        epo.drop_channels(['vEOG', 'hEOG', 'A1', 'A2'])
    except:
        no_dropChan_dyad.append(dyad)

    # Find the channel that belong to each subject
    all_chans = epo.ch_names
    sub2_chan = [x for x in epo.ch_names if x.startswith('2')]
    sub1_chan = [x for x in all_chans if x not in sub2_chan]

    epo_1 = epo.copy().drop_channels(sub2_chan)
    epo_2 = epo.copy().drop_channels(sub1_chan)

    # Remove prefix in chan names of sub2
    replace_ch_name={}
    for i in sub2_chan:
        replace_ch_name[i] = i[1:]
    
    epo_2.rename_channels(replace_ch_name)

    # Now we only keep the channels that are of interest
    selected_ch = set(epo_1.ch_names).difference(set(selected_chans))
    
    epo_1 = epo_1.copy().drop_channels(selected_ch)
    epo_2 = epo_2.copy().drop_channels(selected_ch)

    print("\nVerify equal epoch count: ")
    mne.epochs.equalize_epoch_counts([epo_1, epo_2])

    # print("\nVerify equal channel count: ")
    # ch_to_drop_in_epo_1 = list(set(epo_1.ch_names).difference(epo_2.ch_names))
    # ch_to_drop_in_epo_2 = list(set(epo_2.ch_names).difference(epo_1.ch_names))
    # if len(ch_to_drop_in_epo_1) != 0:
    #     print('Dropping the following channel(s) in epo_1: {}'.format(ch_to_drop_in_epo_1))
    #     epo_1 = epo_1.drop_channels(ch_to_drop_in_epo_1)
    # else:
    #     print('No channel to drop in epo_1.')
    # if len(ch_to_drop_in_epo_2) != 0:
    #     print('Dropping the following channel(s) in epo_2: {}'.format(ch_to_drop_in_epo_2))
    #     epo_2 = epo_2.drop_channels(ch_to_drop_in_epo_2)
    # else:
    #     print('No channel to drop in epo_2.')

    #  Initializing data and storage  #############################################################
    data_inter = np.array([])
    print(epo_1.get_data().shape, epo_2.get_data().shape)###########################
    data_inter = np.array([epo_1.get_data(), epo_2.get_data()]) #, dtype=mne.io.eeglab.eeglab.EpochsEEGLAB) # Deprecation warning

    # Computing analytic signal per frequency band ################################################
    print("- - - - > Computing analytic signal per frequency band ...")
    sampling_rate = epo_1.info['sfreq']
    complex_signal = hypyp.analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)
    np.save("{}W2W-complex_signal/{}_{}_complexsignal.npy".format(
        save_path, dyad, context
    ), complex_signal, allow_pickle=False)

    # Computing frequency- and time-frequency-domain connectivity ################################
    print("- - - - > Computing frequency- and time-frequency-domain connectivity ...")
    result = hypyp.analyses.compute_sync(complex_signal, mode=ibc_metric, epochs_average=True) # (n_freq, 2*n_channels, 2*n_channels)
    np.save("{}W2W-IBC/{}_{}_{}IBC.npy".format(
        save_path, dyad, context, ibc_metric), result, allow_pickle=False)

    results_IBC[dyad] = result
# %%
print('non_epoch_dyads are {}, no_dropChan_dyad are {}'.format(non_epoch_dyads, no_dropChan_dyad))
# %%

for i in results_IBC.keys():
    print(results_IBC[i].shape)

# the shape  (number of freqyency band, selected sensors x 2, selected sensors x 2)

# %%
summarize_IBC = pd.DataFrame(columns=['dyad_id','IBC'])

###################################X
# Do electrods seps    ############X
# Run with power corr  ############X
###################################X

for i in results_IBC.keys():
    summarize_IBC.loc[len(summarize_IBC)] =[i, results_IBC[i].mean()] 

#%%
summarize_IBC.to_csv('{}W2W-IBC_results/IBC_results_{}.csv'.format(
    save_path, context), sep=',')
# %%
