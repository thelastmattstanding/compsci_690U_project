from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
amac_to_idx = {amac: i for i, amac in enumerate(AMINO_ACIDS)}

def remove_unknown_chars(sequence, char_dict):
    cleaned_sequence = []
    for char in sequence:
        if char in char_dict:
            cleaned_sequence.append(char)
    
    return ''.join(cleaned_sequence)

def one_hot_encode(sequence, mode, max_length=None):

    cleaned_sequence = remove_unknown_chars(sequence, amac_to_idx)

    if mode == "mean":
        oh_mean_seq = np.zeros(len(AMINO_ACIDS), dtype=float)
        for char in cleaned_sequence:
            if char in amac_to_idx:
                oh_mean_seq[amac_to_idx[char]] += 1
        oh_mean_seq /= len(cleaned_sequence)
        
        return oh_mean_seq
    
    elif mode == "pad":
        oh_pad_seq = np.zeros((max_length, len(AMINO_ACIDS)), dtype=int)

        for i, char in enumerate(cleaned_sequence[:max_length]):
            oh_pad_seq[i, amac_to_idx[char]] = 1
        
        return oh_pad_seq
    
def label_encode(train_df, test_df):

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["Label"])
    
    train_labels = label_encoder.transform(train_df["Label"])
    test_labels = label_encoder.transform(test_df["Label"])
    
    return train_labels, test_labels

def kmer_encode(sequences, k, min_freq=1):
    cv = CountVectorizer(analyzer='char', ngram_range=(k, k), min_df=min_freq)
    cv.fit(sequences)
    return cv


def extract_features(df, encoding):

    enc_type, enc_mode, max_length = encoding

    if enc_type == "one_hot":
        if enc_mode == "mean":
            oh_mean_seqs = []
            for seq in df["Sequence"]:
                oh_mean_seq = one_hot_encode(seq, enc_mode)
                oh_mean_seqs.append(oh_mean_seq)
                X = np.vstack(oh_mean_seqs)
        
        elif enc_mode == "pad":
            oh_pad_seqs = []
            for seq in df["Sequence"]:
                oh_pad_seq = one_hot_encode(seq, enc_mode, max_length).ravel()
                oh_pad_seqs.append(oh_pad_seq)
            X = np.vstack(oh_pad_seqs)
    else:
        X = enc_mode.transform(df["Sequence"]).toarray()

    y = df["Label"].values
    return X, y


            