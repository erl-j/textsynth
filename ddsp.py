#%%
import torch
import einops
import numpy as np
import julius
import einops
import matplotlib.pyplot as plt
import util
from util import resample_feature

class Reverb(torch.nn.Module):
    def __init__(self,sample_rate,ir_duration, init_ir=None):
        super().__init__()
        self.sample_rate=sample_rate
        self.ir_duration=ir_duration
        ir_samples = int(ir_duration*sample_rate)
        if init_ir is None:
            init_ir = torch.rand(ir_samples,dtype=torch.float32)*0.01
        else:
            assert init_ir.shape[-1]==ir_samples
        self.impulse_response_param=torch.nn.Parameter(init_ir)

    def get_ir(self):
        return torch.cat((torch.zeros(1,dtype=self.impulse_response_param.dtype,device=self.impulse_response_param.device),self.impulse_response_param[1:]))

    def forward(self,signal):
        impulse_response = self.get_ir()
        output = torch.nn.functional.conv1d(padding="same", input=signal.unsqueeze(1), weight=impulse_response[None,None,:]).squeeze(1)  
        return output

def harmonic_synth(f0, harmonic_amplitudes, sample_rate,):

    sample_rate = sample_rate.to(f0.device)

    batch, time, n_harmonics = harmonic_amplitudes.shape
    harm = (torch.arange(n_harmonics, dtype=f0.dtype, device=f0.device)+1)[None,None,:]
    # remove above nyquist
    harmonic_amplitudes = harmonic_amplitudes * (f0 * harm < sample_rate/2)

    phase = torch.cumsum((f0 * 2 * np.pi), dim=1) / sample_rate
    phase = phase * harm

    waveform = torch.sin(phase)
    # remove above nyquist
    waveform = waveform * harmonic_amplitudes
    waveform=waveform.sum(dim=-1)
    return waveform

class HarmonicSynth(torch.nn.Module):
    def __init__(self,sample_rate)->None:
        super().__init__()
        self.sample_rate=torch.tensor(sample_rate)
        # self.harmonic_synth = torch.jit.trace(harmonic_synth, (torch.rand(1,1,1), torch.rand(1,1,1), self.sample_rate))
        self.harmonic_synth = harmonic_synth
    
    def forward(self,f0, harmonic_amplitudes, global_amp, n_samples):
        f0 = resample_feature(f0,n_samples, mode="linear")
        harmonic_amplitudes = resample_feature(harmonic_amplitudes,n_samples, mode="linear")
        global_amp = resample_feature(global_amp,n_samples, mode="linear")
        waveform = self.harmonic_synth(f0, harmonic_amplitudes, self.sample_rate)
        waveform = waveform * global_amp[...,0]
        return waveform

class FilteredNoiseSynthSlow(torch.nn.Module):
    def __init__(self,sample_rate):
        super().__init__()
        self.sample_rate=sample_rate

    def forward(self,band_amplitudes, n_samples):
        # uniform noise
        noise = torch.rand((band_amplitudes.shape[0],n_samples),dtype=band_amplitudes.dtype,device=band_amplitudes.device)*2-1
        # split into bands
        noise_bands = julius.bands.split_bands(noise, n_bands= band_amplitudes.shape[-1], sample_rate=self.sample_rate, fft=True) # True means using julius fftconv isntead of torch conv
        resampled_band_amplitudes = resample_feature(band_amplitudes,n_samples, mode="linear")
        # apply band amplitudes
        noise_bands = [noise_band * band_amp[...,0] for noise_band, band_amp in zip(noise_bands, resampled_band_amplitudes.split(1, dim=-1))]
        # sum bands
        noise = torch.sum(torch.stack(noise_bands, dim=-1), dim=-1)
        return noise


# from https://github.com/acids-ircam/ddsp_pytorch/blob/master/ddsp/core.py
def amp_to_impulse_response(amp, target_size):
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = torch.fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = torch.nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp

# from https://github.com/acids-ircam/ddsp_pytorch/blob/master/ddsp/core.py
def fft_convolve(signal, kernel):
    signal = torch.nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = torch.nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output

# adapted from https://github.com/acids-ircam/ddsp_pytorch/blob/master/ddsp/core.py
class FilteredNoiseSynth(torch.nn.Module):
    def __init__(self,sample_rate):
        super().__init__()
        self.sample_rate=sample_rate
        self.block_size = 128 * sample_rate // 16000
        self.sample_rate = sample_rate

    def forward(self,band_amplitudes, n_samples):
        batch_size, timesteps, n_bands = band_amplitudes.shape
        # uniform noise
        n_blocks = int(np.ceil(n_samples / self.block_size))

        band_amplitudes = resample_feature(band_amplitudes, n_blocks, mode="linear")

        impulse = amp_to_impulse_response(band_amplitudes, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)[...,0]
      
        return noise


def get_string_harmonic_amplitudes(f,n_harmonics,wave_speed,pluck_distance_ratio,pluck_height,measure_distance_ratio):
    """
    f: frequency
    n_harmonics: number of harmonics to calculate
    wave_speed: speed of the wave
    pluck_distance: distance from the pluck to the measurement point
    pluck_height: height of the pluck
    measure_distance: distance from the pluck to the measurement point
    """
    wave_speed = wave_speed[:,None]
    measure_distance_ratio = measure_distance_ratio[:,None]

    n = torch.arange(1,n_harmonics+1, device=f.device)[None,None,:]
    omega = 2*np.pi*f

    string_length = np.pi*wave_speed*(1/omega)

    pluck_height = pluck_height
    pluck_distance_ratio = pluck_distance_ratio

    pluck_distance = pluck_distance_ratio*string_length
    
    # express function in 4 blocks for readability
    a = (2*pluck_height*string_length**2)/((n**2)*(np.pi**2)*pluck_distance*(string_length-pluck_distance))
    b = torch.sin(pluck_distance*n*np.pi/string_length)

    c = torch.sin(measure_distance_ratio*n*np.pi)
    return a*b*c

# model of fourier series of a ideal string 
# see https://www.acs.psu.edu/drussell/Demos/Pluck-Fourier/Pluck-Fourier.html
class StringModel(torch.nn.Module):
    def __init__(self, sample_rate, n_channels, n_harmonics) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.harmonic_synth = HarmonicSynth(sample_rate)
        # create one parameter for each channel
        self.wave_speed_w = torch.nn.Parameter(torch.ones((n_channels))+0.1)
        self.measure_distance_ratio_w = torch.nn.Parameter(torch.ones((n_channels))+10.0)

    def forward(self, frequency,pluck_distance_ratio,pluck_height,n_samples,measure_distance_ratio=None,wave_speed=None, return_harmonic_amplitudes=False):
        # create parameter for the wave speed and the measure distance ratio
        batch, n_channels, n_frames , ft= frequency.shape

        if measure_distance_ratio is None:
            measure_distance_ratio = torch.sigmoid(self.measure_distance_ratio_w)[None,:].repeat(batch,1)
        if wave_speed is None:
            wave_speed = torch.relu(self.wave_speed_w)[None,:].repeat(batch,1)

        # reshape all conditioning to (batch n_channels) ...
        frequency = einops.rearrange(frequency,'b c t f-> (b c) t f')
        pluck_distance_ratio = einops.rearrange(pluck_distance_ratio,'b c t f-> (b c) t f')
        pluck_height = einops.rearrange(pluck_height,'b c t f-> (b c) t f')

        measure_distance_ratio = einops.rearrange(measure_distance_ratio,'b c -> (b c)')[...,None]
        wave_speed = einops.rearrange(wave_speed,'b c-> (b c)')[...,None]

        # calculate the amplitudes of the harmonics
        harmonic_amplitudes = get_string_harmonic_amplitudes(frequency,self.n_harmonics,wave_speed,pluck_distance_ratio,pluck_height,measure_distance_ratio)

        # reshape back to (batch, n_channels, n_frames, n_harmonics)
        audio = self.harmonic_synth(frequency,harmonic_amplitudes,torch.ones_like(frequency),n_samples)
        audio = einops.rearrange(audio,'(b c) t-> b c t',c=n_channels)

        if return_harmonic_amplitudes:
            return audio,harmonic_amplitudes
        else:
            return audio
# %%
