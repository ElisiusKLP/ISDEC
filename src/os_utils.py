
def mne_to_dict(epochs, subject_id):
    x = epochs.get_data()
    y = epochs.events[:, -1]
    sfreq = epochs.info['sfreq']

    return {
        'x': x,
        'y': y,
        'sfreq': sfreq,
        'subject_id': subject_id
    }