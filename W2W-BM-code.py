# @Author: Franck Portoeus <franck.porteous@protonmail.com>
# Last update: Dec 14th 2022

#%% Imports
import hypyp

import mne
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
import seaborn as sns

#%% Defining OS paths
data_path    = "../../W2W-Data/W2W-EEG/W2W-Brittany-Data/"
save_path    = "../../W2W-save/"

#%% Ploting options
# run this for interactive plots
# %matplotlib qt
# run this for static plots
# %matplotlib inline 

#%% Listing the files to work with and removing uncecessary files
all_file = os.listdir(data_path) 
try:
    all_file.remove('.DS_Store')
except:
    pass

#%% Setting up the study's parameters (chs, freqs, metrics, ...)

# Also, for the analysis metrics, first priority is PLV. 
# Let's do 8-9 hz in P3, Pz, P4, C3, C4, P03, P04 - can we 
# keep electrodes separate for now? If that's too challenging, 
# we can prioritize P3 and P4. 

# Second priority would be Power Correlation in those same electrodes. 

ibc_metric = 'pow_corr' # 'plv' Hypyp documentation https://hypyp.readthedocs.io/en/latest/
selected_chans = ['P3', 'Pz', 'P4', 'C3', 'C4', 'P03', 'P04']
n_ch = len(selected_chans)
freq_bands = {'Theta': [8.0, 9.0]}
nb_freq_band = len(freq_bands.keys())

#%% Create the dict where the IBC will be stored

results_IBC = {}

# If we want to setup so the resutls are stored in a np.array we can do the folowing:
# results_IBC = np.zeros([nb_of_dyad, nb_freq_band, n_ch*2, n_ch*2], dtype=np.float32)

# Create a list to keep a record of the channel order 
ch_order = []

#%% Initializing droplog df for files with continuous data and those who don't have 'vEOG', 'hEOG', 'A1', 'A2'
non_epoch_dyads  = []
no_dropChan_dyads = []


# Define a function to slice IBC matrix
def get_ch_idx(soi:list(), n_ch:int(), quadrant:str()):
    '''
    Returns a tool to slice a IBC matrix by providing channel idx.
    CAREFUL: This versions assumes that you're using an IBC metric
    that is non-directional.
    '''
    assert quadrant in ['inter', 'intra_A', 'intra_B'], "Quadrant is wrong"

    cut1, cut2 = soi, soi

    if quadrant == 'inter':
        cut2 = [x+n_ch for x in cut2]
    elif quadrant == 'intra_A':
        pass # The idx already indicate these locs
    elif quadrant == 'intra_B':
        cut1 = [x+n_ch for x in cut1]
        cut2 = [x+n_ch for x in cut2]   
    
    cut = np.ix_(cut1, cut2)
    return cut

#%% Set up context 
context = ['form' ,'book', 'puzzle', 'movie']

#%% Run the analysis 

for cont in context:

    # Get unique dyad number for a given context
    dyad_nb=[]
    for i in all_file:
        if i.split('_')[2] == cont:
            dyad_nb.append(i.split('_')[1])
        else:
            pass
    dyad_nb = np.unique(dyad_nb)
    nb_of_dyad = len(dyad_nb)

    for dyad in dyad_nb:

        try: 
            # First, we try to load the epoch file.
            # If the file is continuous (ie raw) then it'll throw an error, 
            # in that case, we store the dyad number in the non_epoch_dyads list
            # and don't proceed with the file. 
            epo = mne.read_epochs_eeglab('{}W2W_{}_{}_hpfilt_chlocs_childinterp_parentinterp_binned_runICA_componentsremoved_childAR_parentAR_selectedepochs.set'.format(
            data_path, dyad, cont))

            print('{}W2W_{}_{}_hpfilt_chlocs_childinterp_parentinterp_binned_runICA_componentsremoved_childAR_parentAR_selectedepochs.set'.format(
            data_path, dyad, cont))
            

            try:
                # Second, we try to drop non-necessary channels 
                # If the EEG doesn't has them it'll throw an error, 
                # in that case, we store the dyad number in the 
                # no_dropChan_dyads list and don't proceed with the file. 

                epo.drop_channels(['vEOG', 'hEOG', 'A1', 'A2'])

                #######################
                # If no error occured when reading the EEGLAB file and that the rejection
                # of 'vEOG', 'hEOG', 'A1', 'A2' was successful, then we procede to the IBC analysis.

                sampling_rate = epo.info['sfreq']

                #######################
                # Separate the dual-EEG file into 2 mne.epoch object (one for each participant)
                all_chans = epo.ch_names                                       # List the chanel name in dual-eeg
                sub2_chan = [x for x in epo.ch_names if x.startswith('2')]     # Find chs from sub2
                sub1_chan = [x for x in all_chans if x not in sub2_chan]       # Find chs from sub1

                epo_1 = epo.copy().drop_channels(sub2_chan)  # Make a copy of dual-eeg and drop other participant's chs
                epo_2 = epo.copy().drop_channels(sub1_chan)  # Idem

                # Remove prefix in chan names of sub2
                replace_ch_name={}                      # Create what will be the remplacement dict
                for i in sub2_chan:                     
                    replace_ch_name[i] = i[1:]          # Define the key as the old name and value as new (stripped of '2') name for chs
                epo_2.rename_channels(replace_ch_name)  # Change the chs name in sub2

                # Now we only keep the channels that are of interest
                non_selected_ch = set(epo_1.ch_names).difference(set(selected_chans)) # Find the chs that'll have to be dropped
                
                epo_1 = epo_1.copy().drop_channels(non_selected_ch)                   # Drop the chs
                epo_2 = epo_2.copy().drop_channels(non_selected_ch)                   # idem

                # Sanity check: The epoch count in each mne.epoch object must be the same
                print("\nVerify equal epoch count: ")
                mne.epochs.equalize_epoch_counts([epo_1, epo_2])

                ch_order = epo_1.ch_names

                ##################### 
                # The commented block below would be used to reject chs in
                # in a participant that are not present in the other, and vice versa.
                ##################### 

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
                data_inter = np.array([epo_1.get_data(), epo_2.get_data()]) #, dtype=mne.io.eeglab.eeglab.EpochsEEGLAB) # Deprecation warning

                # Computing analytic signal per frequency band ################################################
                print("- - - - > Computing analytic signal per frequency band ...")
                complex_signal = hypyp.analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)
                np.save("{}W2W-complex_signal/{}_{}_complexsignal.npy".format(
                    save_path, dyad, cont),
                    complex_signal, allow_pickle=False)  
                    # Save df as .nyp

                # Computing frequency- and time-frequency-domain connectivity ################################
                print("- - - - > Computing frequency- and time-frequency-domain connectivity ...")
                result = hypyp.analyses.compute_sync(complex_signal, mode=ibc_metric, epochs_average=True)
                # the np.array "result" is of shape (n_freq, 2*n_channels, 2*n_channels)
                np.save("{}W2W-IBC/{}_{}_{}.npy".format(
                    save_path, dyad, cont, ibc_metric),
                    result, allow_pickle=False)          
                    # Save df as .nyp

                results_IBC[dyad] = result

                #del epo, epo_1, epo_2, sub1_chan, sub2_chan, replace_ch_name, data_inter, complex_signal, result, dyad_nb
                
                

            except:
                no_dropChan_dyads.append(dyad)
        except:
            non_epoch_dyads.append(dyad)

    # Save the droplog
    with open('{}W2W-IBC_droplog/{}_droplog.txt'.format(save_path, cont), 'w') as f:
        f.write('In context {}, non_epoch_dyads are {}, no_dropChan_dyads are {}'.format(cont, non_epoch_dyads, no_dropChan_dyads))
    f.close()

    # Check the shape of the results (sanity check)
    for i in results_IBC.keys():
        print(results_IBC[i].shape)
        # the shape should be (number of frequency band, selected sensors x 2, selected sensors x 2)

    # Store data

    # Slice teh IBC matrix to get IBC between pairs of the same sensors
        # E.g., FP1 of sub-1 with FP1 of sub-2 

    # Create an empty dataframe to store the IBC values
    summarize_IBC = pd.DataFrame(columns=ch_order)

    # for dyad in ['2023']:
    for dyad in results_IBC.keys():
        for chan_idx, chan_name in enumerate(ch_order):
            
            # Here we give the index of the sensor(s) of interest
            cut = get_ch_idx(soi=[chan_idx], n_ch=len(ch_order), quadrant='inter')
            summarize_IBC.loc[dyad, chan_name] = results_IBC[dyad][0][cut[0], cut[1]].mean()

    # Add a mean column (do that before adding the dyad ID in a column..)
    summarize_IBC['IBC_average'] = summarize_IBC.mean(axis=1)

    # Add the index as a column 
    summarize_IBC.reset_index(inplace=True)
    summarize_IBC = summarize_IBC.rename(columns = {'index':'Dyads'})

    summarize_IBC

    # Save the data
    summarize_IBC.to_csv('{}W2W-IBC_results/IBC_results_{}_{}.csv'.format(
        save_path, cont, ibc_metric), sep=',')
    print
# %%