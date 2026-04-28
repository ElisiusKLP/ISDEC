import mne
from pathlib import Path
from mne.preprocessing import ICA
from mne_icalabel import label_components

#Load mne objects

mne_dir = Path("data/derivatives/mne")

def preprocess (input_dir):
    all_files = list(data_dir.glob('*.fif'))

    for file in all_files:
        subject_file = mne.read_epochs(file)

        copy_file = subject_file.copy().filter(1.0,100)

        #ICA should be done before baseline correcting

        #ica = ICA(n_components=20, max_iter="auto", random_state=2001)
       
        #ica.fit(copy_file)

        #ic_labels = label_components(subject_file,ica, 'iclabel')

        #Fnd labels, pass to ica.exclude, then apply ica

        subject_file.set_eeg_reference(ref_channels="average")

        #possibly use AutoReject to mark badchannels

# https://mne.tools/mne-icalabel/stable/generated/examples/00_iclabel.html#sphx-glr-generated-examples-00-iclabel-py 
