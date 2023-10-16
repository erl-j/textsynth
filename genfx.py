#%%
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torchsynth.synth import Voice,SynthConfig
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.cluster import KMeans
from tqdm import tqdm
import IPython.display as display
import pandas as pd
import laion_clap
import librosa
from pedalboard import (
    Pedalboard,
    Chorus,
    Reverb,
    PitchShift,
    Delay,
    Compressor,
    Distortion,
    LadderFilter,
    Mix,
    Gain,
)


SAMPLE_RATE=48000

# Run on the GPU if it's available
if torch.cuda.is_available():
    device = torch.device("cuda:3")
else:
    device = torch.device("cpu")

# disable gradient calculation
torch.set_grad_enabled(False)

def play(a):
    display.display(display.Audio(a, rate=SAMPLE_RATE))


class EffectChain:
    def __init__(self):
        self.parameters = [
            "drive_db",
            "reverb_room_size",
            "reverb_damping",
            "reverb_mix",
            "chorus_rate",
            "chorus_depth",
            "chorus_mix",
            "delay_time",
            "delay_feedback",
            "delay_mix",
            "compressor_threshold",
            "compressor_ratio",
            "octave_up_mix",
            "octave_down_mix",
            "low_pass_cutoff",
            "high_pass_cutoff",
        ]

    def get_n_parameters(self):
        return len(self.parameters)

    def tensor2pedalboard(self, tensor):

        # parameter 2 value
        p2v = dict(zip(self.parameters, tensor))

        board = Pedalboard(
            [
                Chorus(
                    rate_hz=p2v["chorus_rate"] * 2,
                    depth=p2v["chorus_depth"],
                    mix=p2v["chorus_mix"],
                ),
                Reverb(
                    room_size=p2v["reverb_room_size"],
                    damping=p2v["reverb_damping"],
                    dry_level=1 - p2v["reverb_mix"],
                    wet_level=p2v["reverb_mix"],
                ),
                Delay(
                    delay_seconds=p2v["delay_time"],
                    feedback=p2v["delay_feedback"],
                    mix=p2v["delay_mix"],
                ),
                Compressor(
                    threshold_db=-p2v["compressor_threshold"] * 10,
                    ratio=1.0 + p2v["compressor_ratio"] * 100.0,
                ),
                Distortion(drive_db=p2v["drive_db"] * 50),
                Mix(
                    [
                        Pedalboard(
                            [
                                PitchShift(semitones=12),
                                Gain(gain_db=-40 + 40 * p2v["octave_up_mix"]),
                            ]
                        ),
                        Pedalboard(
                            [
                                PitchShift(semitones=-12),
                                Gain(gain_db=-40 + 40 * p2v["octave_down_mix"]),
                            ]
                        ),
                        Gain(gain_db=0),
                    ]
                ),
                LadderFilter(
                    mode=LadderFilter.Mode.HPF12,
                    cutoff_hz=p2v["high_pass_cutoff"] * 16000,
                ),
                LadderFilter(
                    mode=LadderFilter.Mode.LPF12,
                    cutoff_hz=p2v["low_pass_cutoff"] * 16000,
                ),
            ]
        )

        return board

    def __call__(self, audio, tensor):
        board = self.tensor2pedalboard(tensor)
        return board(audio, sample_rate=SAMPLE_RATE)


CLIP_DURATION=1
BATCH_SIZE=40
N_PARENTS=64

N_SAVED_PER_TARGET=16

MUTATION_RATE=0.01
TEMPERATURE=2.0

MIDI_F0=53


# initialize population
activation = lambda x: 0.0000001 + 0.9999 * (torch.cos(x * np.pi * 2) + 1) / 2
effect_chain = EffectChain()


source_path = "./data/nylon.wav"
source_audio = librosa.load(source_path, sr=SAMPLE_RATE, duration=CLIP_DURATION)[0]

dummy_p=torch.randn((BATCH_SIZE, effect_chain.get_n_parameters()))

clap_model = laion_clap.CLAP_Module(enable_fusion=True, device=device)
clap_model.load_ckpt() # download the default pretrained checkpoint.

def embed_audio(audio_data):
    audio_embed = clap_model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
    return audio_embed

def embed_text(text_data):
    text_embed = clap_model.get_text_embedding(text_data, use_tensor=True)
    return text_embed

# random p
p = torch.rand(dummy_p.shape).to(device)*2*np.pi

def mutate(p):
    #mask=(torch.rand(p.shape,device=p.device)<MUTATION_RATE)
    #p = mask*torch.rand(p.shape,device=p.device)*2*np.pi+(~mask)*p
    p += torch.randn(p.shape,device=p.device)*MUTATION_RATE
    return p

def crossover(p1,p2):
    mask = torch.rand(p1.shape,device=p1.device)>0.5
    return p1*mask+p2*(~mask)

#%%
PROMPT = "a bass guitar"
# random p
p = torch.rand(dummy_p.shape).to(device)*2*np.pi
records = []
zt = embed_text([PROMPT,PROMPT])[:1]

generation=0
while True:
    # synthesize 
    audio = [
        effect_chain(source_audio, activation(p[i]))
        for i in range(BATCH_SIZE)
    ]
    # turn into tensor
    audio = torch.tensor(audio).float().to(device)
    # peak normalize each sample
    peaks = torch.max(torch.abs(audio),dim=1,keepdim=True)[0]
    audio = audio/peaks
    # embed audio
    za = embed_audio(audio)

    # novelty search

    # 
    # get fitness by measuring similarity to target
    similarity = torch.nn.functional.cosine_similarity(za,zt)
    for b in range(BATCH_SIZE):
        records.append({"generation":generation,"similarity":similarity[b].item(),"p":p[b].detach().cpu().numpy()})

    fitness = torch.softmax(similarity/TEMPERATURE,dim=0)

    p1 = p[torch.multinomial(fitness,BATCH_SIZE,replacement=True)]
    p2 = p[torch.multinomial(fitness,BATCH_SIZE,replacement=True)]

    p = crossover(p1,p2)

    p = mutate(p)

    generation+=1
    
    if generation%1==0:

            # clear output
        clear_output(wait=True)
        # plot sorted fitness
        plt.plot(torch.sort(fitness).values.detach().cpu().numpy())
        plt.show()

        # show scatter plot of similarity
        sns.scatterplot(data=pd.DataFrame(records),x="generation",y="similarity",alpha=0.5)
        plt.show()
        # play audio of samples sorted by similarity
        play(audio[torch.argsort(-similarity)].flatten().detach().cpu().numpy())


# %%
