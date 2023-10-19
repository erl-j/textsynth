import einops
import librosa
import numpy as np
import torch
from IPython.display import Audio, display

DB_RANGE = -80.0

EPS=1e-7


def features_to_notes(features):
# extract list of notes from midi pitch and velocity. 
# Look at changes in velocity or midi pitch to determine note on/off
    notes = []
    for string_idx in range(6):
        note = {}
        for frame_idx in range(features["midi_pitch"].shape[-1]):
            is_note_already_on = "midi_pseudo_velocity" in note
            # case 1 - note is already on
            if is_note_already_on:
                # case 1a - current is same as previous
                if features["midi_pitch"][0, string_idx, frame_idx] == note["midi_pitch"] and features["midi_pseudo_velocity"][0, string_idx, frame_idx] == note["midi_pseudo_velocity"]:
                    pass
                else:
                    # case 1b - current is different from previous
                    note["offset_frame"] = frame_idx
                    notes.append(note)
                    note = {}
                    is_note_already_on = False
            # case 2 - note is not already on
            if not is_note_already_on:
                # if current is not zero, turn on note
                if features["midi_pitch"][0, string_idx, frame_idx] != 0:
                    note["onset_frame"] = frame_idx
                    note["string_index"] = string_idx
                    note["midi_pitch"] = features["midi_pitch"][0, string_idx, frame_idx]
                    note["midi_pseudo_velocity"] = features["midi_pseudo_velocity"][0, string_idx, frame_idx]
                    is_note_already_on = True
            # if last frame, turn off note
            if frame_idx == features["midi_pitch"].shape[-1]-1 and is_note_already_on:
                note["offset_frame"] = features["midi_pitch"].shape[-1]
                notes.append(note)
    return notes

def notes_to_features(notes, conditioning_shape):
    features = {}
    features["midi_pitch"] = torch.zeros(conditioning_shape)
    features["midi_pseudo_velocity"] = torch.zeros(conditioning_shape)+1
    #reconstruct midi pitch and velocity from notes
    for note in notes:
        features["midi_pitch"][0, note["string_index"], note["onset_frame"]:note["offset_frame"]] = note["midi_pitch"]
        features["midi_pseudo_velocity"][0, note["string_index"], note["onset_frame"]:note["offset_frame"]] = note["midi_pseudo_velocity"]
    return features


def weights_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform(m.weight.data)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)

def gaussian_window(n_steps, step_index, sigma):
    loc = step_index / n_steps
    linspace = torch.linspace(0, 1, n_steps)
    out = torch.exp(-0.5 * (linspace - loc)**2 / sigma**2)
    # normalize
    out = out / out.sum()
    return out

def viterbi(input_probs, proximity_prior_sigma=1.0):
    # if 3 dimensional, render each batch element separately
    if len(input_probs.shape) == 3:
        return torch.stack([_viterbi(input_probs[i], proximity_prior_sigma) for i in range(input_probs.shape[0])])
    elif len(input_probs.shape) == 4:
        batch, voice, t, f = input_probs.shape
        probs = einops.rearrange(input_probs, "b v t f -> (b v) t f")
        out = torch.stack([_viterbi(probs[i], proximity_prior_sigma) for i in range(probs.shape[0])])
        return einops.rearrange(out, "(b v) t f -> b v t f", b=batch, v=voice, t=t, f=f)
    
def _viterbi(input_probs, proximity_prior_sigma):
    """
    Perform Viterbi decoding on a sequence of probabilities.
    """
    # transition_probs: (t, n_states, n_states)
    seq_len, n_states = input_probs.shape

    proximity_bias = torch.zeros(n_states, n_states)

    for i in range(n_states):
        # create a gaussian window around the current state
        proximity_bias[i] = gaussian_window(n_states, i, proximity_prior_sigma)

    # factor in proximity bias to the transition probs
    transition_log_probs = torch.log(input_probs[:,None,:]) + torch.log(proximity_bias[None,:,:])

    viterbi_probs = torch.zeros((seq_len, n_states))
 
    viterbi_paths = torch.zeros((seq_len, n_states), dtype=torch.int64)

    # initialize the viterbi probs for the first step
    viterbi_probs[0] = transition_log_probs[0, 0]

    # initialize the viterbi paths for the first step
    viterbi_paths[0] = torch.arange(n_states)

    # iterate over the remaining steps
    for t in range(1, seq_len):
        # compute the max probs and paths for the current step
        viterbi_probs[t], viterbi_paths[t] = torch.max(viterbi_probs[t-1].unsqueeze(-1) + transition_log_probs[t], dim=0)
    
    # backtrack to find the best path
    best_path = torch.zeros(seq_len, dtype=torch.int64)
    best_path[-1] = torch.argmax(viterbi_probs[-1])
    for t in range(seq_len-2, -1, -1):
        best_path[t] = viterbi_paths[t+1, best_path[t+1]]

    out= torch.nn.functional.one_hot(best_path, num_classes=n_states).float()
    return out    

def forward_fill_midi_pitch(midi_pitch):
    for string_index in range(6):
        last_nonzero = 0
        # iterate in reverse order
        for frame_index in range(midi_pitch.shape[1]-1, -1, -1):
            if midi_pitch[string_index, frame_index] != 0:
                last_nonzero = midi_pitch[string_index, frame_index]
            else:
                midi_pitch[string_index, frame_index] = last_nonzero
        
        # iterate in forward order
        last_nonzero = 0
        for frame_index in range(midi_pitch.shape[1]):
            if midi_pitch[string_index, frame_index] != 0:
                last_nonzero = midi_pitch[string_index, frame_index]
            else:
                midi_pitch[string_index, frame_index] = last_nonzero
    open_string_midi_pitch = [40, 45, 50, 55, 59, 64]
    for string_index in range(6):
        if torch.sum(midi_pitch[string_index]) == 0:
            midi_pitch[string_index] = open_string_midi_pitch[string_index]
    return midi_pitch   


class Quantizer:
    def __init__(self, values, n_bins):
        self.values = values
        values = np.sort(values)
        self.min_value = np.min(values)
        self.max_value = np.max(values)
        self.n_bins = n_bins

    def quantize(self, x):
        # clip values
        x = torch.clamp(x, self.min_value, self.max_value)
        return linear_quantize(x, self.min_value, self.max_value, self.n_bins)
    
    def dequantize(self, x):
        return linear_dequantize(x, self.min_value, self.max_value, self.n_bins)


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def linear_quantize(value, min_value, max_value, n_bins):
    # remove last dimension of value
    value = value.squeeze(-1)
    return torch.round((value - min_value) / (max_value - min_value) * (n_bins - 1)).long()

def linear_dequantize(bin, min_value, max_value, n_bins):
    return (bin / (n_bins - 1) * (max_value - min_value) + min_value).float()

def se(targets,predictions):
    return (targets-predictions)**2

def scale_db(db):
  """Scales [-DB_RANGE, 0] to [0, 1]."""
  return (db / DB_RANGE) + 1.0

def inv_scale_db(db_scaled):
  """Scales [0, 1] to [-DB_RANGE, 0]."""
  return (db_scaled - 1.0) * DB_RANGE

def hz_to_midi(f):
    if isinstance(f, int) or isinstance(f, float) or isinstance(f, np.ndarray):
        f = np.clip(f, EPS, np.inf)
        m = 12 * np.log2(f / 440) + 69
    else:
        f = torch.clip(f, EPS, np.inf)
        m = 12 * torch.log2(f / 440) + 69
    return m

def hz_to_midi_noclip(f):
    if isinstance(f, int) or isinstance(f, float) or isinstance(f, np.ndarray):
        m = 12 * np.log2(f / 440) + 69
    else:
        m = 12 * torch.log2(f / 440) + 69
    return m
    
def midi_to_hz(m):
    f= 440 * 2 ** ((m - 69) / 12)
    f= torch.clip(f, EPS, np.inf)
    return f

def midi_to_unit(m, midi_min, midi_max, clip=True):
    midi = torch.clip(m, midi_min, midi_max) if clip else m
    return (midi - midi_min) / (midi_max - midi_min)

def unit_to_midi(u, midi_min,midi_max, clip=True):
    unit = torch.clip(u, 0, 1) if clip else u
    return unit * (midi_max - midi_min) + midi_min

def hz_to_unit(f,hz_min,hz_max,clip=True):    
    return midi_to_unit(hz_to_midi(f), hz_to_midi(hz_min), hz_to_midi(hz_max), clip)

def unit_to_hz(u, hz_min, hz_max, clip=True):
    return midi_to_hz(unit_to_midi(u, hz_to_midi(hz_min), hz_to_midi(hz_max), clip))

def resample_feature(x, out_samples, mode):
    """Resamples a feature to a new sample rate.

    Args:
        x: A torch.Tensor with 2 dims (batch, channels), 3 dims (batch, channels, samples) or 4 dims (batch, channels, samples, ft).
        out_samples: The target sample rate.
        dim: The dimension to resample.
    Returns:
        A torch.Tensor of shape (batch, channels, out_sample).
    """
    n_input_dims = len(x.shape)

    if n_input_dims < 4:
        return resample_feature_2d_3d(x, out_samples,mode)

    if n_input_dims == 4:
        batch, channel, time, feature = x.shape
        # reshape to 3d
        x = einops.rearrange(x, "batch channel time feature -> (batch channel feature) time")
        x = resample_feature_2d_3d(x, out_samples,mode)
        # reshape back to 4d
        x = einops.rearrange(x, "(batch channel feature) time -> batch channel time feature", batch=batch, channel=channel)
        return x
    

def resample_feature_2d_3d(x, out_samples, mode):
    # TODO: ugly code, clean up

    n_input_dims = len(x.shape)
    # if channel dim is missing, add it
    if n_input_dims < 3:
        x = x[:,...,None]
    # set x to last channel for it to work with torch.nn.functional.interpolate
    x = x.transpose(1,-1)
    # resample to target sample rate
    x = torch.nn.functional.interpolate(x, out_samples, mode=mode)
    # set x back to first channel
    x = x.transpose(1,-1)
    # remove channel dim if it was missing
    if n_input_dims < 3:
        x = x[...,0]
    # reshape into original batch and channel dims
    return x

def play_audio(waveform, sample_rate):
    display(Audio(waveform, rate=sample_rate))

def max_pool_resample(x, out_samples):
    # resample last dimension to out_samples by taking max
    return torch.nn.functional.max_pool1d(x, out_samples)





class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, ds, n_repeats, item_idx_to_repeat):
        self.ds = ds
        self.n_repeats = n_repeats
        self.item_idx_to_repeat = item_idx_to_repeat
    def __len__(self):
        return self.n_repeats
    def __getitem__(self, idx):
        return self.ds[self.item_idx_to_repeat]
    
def fold(x):
    ''' 
    Folds channel dimension into batch dimension.
    '''
    if len(x.shape) == 3:
        return einops.rearrange(x, "batch channel time -> (batch channel) time")
    if len(x.shape) == 4:
        return einops.rearrange(x, "batch channel time ft -> (batch channel) time ft")

def unfold(x, n_channels):
    '''
    Unfolds channel dimension from batch dimension.
    '''
    if len(x.shape) == 2:
        return einops.rearrange(x, "(batch channel) time -> batch channel time", channel=n_channels)
    if len(x.shape) == 3:
        return einops.rearrange(x, "(batch channel) time ft -> batch channel time ft", channel=n_channels)

class FakeEpochDataset(torch.utils.data.Dataset):
    '''
    Wraps a dataset to and says __len__ is n_samples.
    Under hood it sequentially draws from the dataset and starts over when it reaches the end.
    '''
    def __init__(self, ds, n_samples):
        self.ds = ds
        self.n_samples = n_samples
        self.internal_idx = 0
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        item = self.ds[self.internal_idx]
        self.internal_idx = (self.internal_idx + 1) % len(self.ds)
        return item

def convert_dtype(dict,from_dtype_to_dtype):
    for key, value in dict.items():
        if isinstance(value,torch.Tensor) and value.dtype in from_dtype_to_dtype:
            dict[key] = value.to(from_dtype_to_dtype[value.dtype])
    return dict