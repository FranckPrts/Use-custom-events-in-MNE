#!/usr/bin/env python
# coding=utf-8
# Original @author Joseph Rudoler <https://github.com/jrudoler>
# Adapted by Franck Porteous <franck.porteous@proton.me>

from logging import Logger as logger
from nolds import hurst_rs
import numpy as np
import os
import scipy.stats as ss
print('imported')
def detect_bad_channels(eeg, save_dir, ignore=None):
# def detect_bad_channels(eeg, save_dir, basename, ephys_dir, ignore=None):
        """
        Runs several bad channel detection tests, records the test scores in a TSV file, and saves the list of bad
        channels to a text file. The detection methods are as follows:
        
        1) Log-transformed variance of the channel. The variance is useful for identifying both flat channels and
        extremely noisy channels. Because variance has a log-normal distribution across channels, log-transforming the
        variance allows for more reliable detection of outliers.
        
        2) Hurst exponent of the channel. The Hurst exponent is a measure of the long-range dependency of a time series.
        As physiological signals consistently have similar Hurst exponents, channels with extreme deviations from this
        value are unlikely to be measuring physiological activity.
        
        A third method used to look for high voltage offset from the reference channel. This corresponds to the
        electrode offset screen in BioSemi's ActiView, and can be used to identify channels with poor connection to the
        scalp. The percent of the recording during which the voltage offset exceeds 40 mV would be calculated for each
        channel. Any channel that spent more than 25% of the total duration of the partition above this offset
        threshold would be marked as bad. Unfortunately, this method does not work after high-pass filtering, due to the
        centering of each channel around 0. Because sessions are now partitioned for bad channel detection, high pass
        filtering must be done prior to all bad channel detection, resulting in this method no longer working.
        
        Note that high-pass filtering is required prior to calculating the variance and Hurst exponent of each channel,
        as baseline drift will artificially increase the variance and invalidate the Hurst exponent.
        
        Following bad channel detection, two bad channel files are created. The first is a file named
        <eegfile_basename>_bad_chan<index>.txt (where index is the partition number for that part of the session), and
        is a text file containing the names of all channels that were identifed as bad. The second is a tab-separated
        values (.tsv) file called <eegfile_basename>_bad_chan_info<index>.tsv, which contains the actual detection
        scores for each EEG channel.
        
        :param eeg: An mne Raw object containing the EEG data to run bad channel detection on.
        :param index: The partition number of this part of the session. Used so that each parallel job writes a
            different bad channel file.
        :param basename: The basename of the EEG recording. Used for naming bad channel files in a consistent manner.
        :param ephys_dir: The path to the ephys directory for the session.
        :param ignore: A boolean array indicating whether each time point in the EEG signal should be excluded/ignored
            during bad channel detection.
        :return: A list containing the string names of each bad channel.
        
        """
        # logger.debug('Identifying bad channels for part %i of %s' % (index, basename))

        # Set thresholds for bad channel criteria (see docstring for details on how these were optimized)
        low_var_th = -3  # If z-scored log variance < -3, channel is most likely flat
        high_var_th = 3  # If z-scored log variance > 3, channel is likely too noisy
        hurst_th = 3  # If z-scored Hurst exponent > 3, channel is unlikely to be physiological
        n_chans = eeg._data.shape[0]

        """
        # Deprecated Method 1: Percent of samples with a high voltage offset (>40 mV) from the reference channel
        # Does not work after high-pass filtering
        offset_th = .04  # Samples over ~40 mV (.04 V) indicate poor contact with the scalp (BioSemi only)
        offset_rate_th = .25  # If >25% of the recording partition has poor scalp contact, mark as bad (BioSemi only)
        if filetype == '.bdf':
            if ignore is None:
                ref_offset = np.mean(np.abs(eeg._data) > offset_th, axis=1)
            else:
                ref_offset = np.mean(np.abs(eeg._data[:, ~ignore]) > offset_th, axis=1)
        else:
            ref_offset = np.zeros(n_chans)
        """

        # Method 1: High or low log-transformed variance
        if ignore is None:
            var = np.log(np.var(eeg._data, axis=1))
        else:
            var = np.log(np.var(eeg._data[:, ~ignore], axis=1))
        zvar = ss.zscore(var)

        # Method 2: High Hurst exponent
        hurst = np.zeros(n_chans)
        for i in range(n_chans):
            if ignore is None:
                hurst[i] = hurst_rs(eeg._data[i, :])
            else:
                hurst[i] = hurst_rs(eeg._data[i, ~ignore])
        zhurst = ss.zscore(hurst)

        # Identify bad channels using optimized thresholds
        bad = (zvar < low_var_th) | (zvar > high_var_th) | (zhurst > hurst_th)
        badch = np.array(eeg.ch_names)[bad]


# TO ADAPT



        # # Save list of bad channels to a text file
        # badchan_file = os.path.join(ephys_dir, basename + '_bad_chan%i.txt' % index)
        # np.savetxt(badchan_file, badch, fmt='%s')
        # os.chmod(badchan_file, 0o644)

        # # Save a TSV file with extended info about each channel's scores
        # badchan_file = os.path.join(ephys_dir, basename + '_bad_chan_info%i.tsv' % index)
        # with open(badchan_file, 'w') as f:
        #     f.write('name\tlog_var\thurst\tbad\n')
        #     for i, ch in enumerate(eeg.ch_names):
        #         f.write('%s\t%f\t%f\t%i\n' % (ch, var[i], hurst[i], bad[i]))
        # os.chmod(badchan_file, 0o644)

        return badch.tolist()