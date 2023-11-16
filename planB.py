#%%
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchsynth.synth import Voice,SynthConfig
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
import IPython.display as display
import pandas as pd
import laion_clap
import soundfile as sf
import einops
from fast_pytorch_kmeans import KMeans
from texts import prompts

SAMPLE_RATE=48000

# Run on the GPU if it's available
if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

def play(a):
    display.display(display.Audio(a, rate=SAMPLE_RATE))

class SynthWrapper():
    def __init__(self,synth):
        self.synth=synth
        self.dummy_parameter_dict=self.synth.get_parameters()

    def parameterdict2tensor(self,parameters):
        out=[]
        for p in parameters.values():
            out.append(p.data)
        return torch.stack(out,dim=-1)

    def tensor2parameterdict(self,tensor):
        parameter_dict = self.dummy_parameter_dict.copy()
        for i,key in enumerate(parameter_dict.keys()):
            parameter_dict[key].data=tensor[:,i]
        return parameter_dict

    def from_0to1(self,parameterdict):
        for key in parameterdict.keys():
            parameterdict[key].data=parameterdict[key].from_0to1()
        return parameterdict

    def synthesize(self,tensor,MIDI_F0=None):
        with torch.no_grad():
            parameter_dict=self.tensor2parameterdict(tensor)
            #parameter_dict=self.from_0to1(parameter_dict)
            if MIDI_F0 is not None:
                parameter_dict[('keyboard', 'midi_f0')].data=parameter_dict[('keyboard', 'midi_f0')].data*0.0+MIDI_F0/127.0
            self.synth.freeze_parameters(parameter_dict)
        return self.synth.output()

    def get_number_of_parameters(self,):
        return len(self.dummy_parameter_dict.keys())

    def get_parameter_tensor(self,):
        return self.parameterdict2tensor(self.synth.get_parameters())
    
class CLAPWrapper():
    def __init__(self):
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=True, device=device)
        self.clap_model.load_ckpt() # download the default pretrained checkpoint.

    def embed_audio(self,audio_data):
        audio_embed = self.clap_model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
        return audio_embed

    def embed_text(self,text_data):
        text_embed = self.clap_model.get_text_embedding(text_data, use_tensor=True)
        return text_embed
    

CLIP_DURATION=1
BATCH_SIZE=60
MIDI_F0=53
CLAP_Z_SIZE = 512


#%%
config = SynthConfig(batch_size=BATCH_SIZE,sample_rate=SAMPLE_RATE,reproducible=False,buffer_size_seconds=CLIP_DURATION)
synth = SynthWrapper(Voice(config).to(device))
dummy_p=synth.get_parameter_tensor()
clap = CLAPWrapper()

N_SOUNDS = 100_000

records = []

for gen in tqdm(range(10000)):
    with torch.no_grad():
        # random sample p
        p = torch.rand(dummy_p.shape).to(device)
        # synthesize 
        audio = synth.synthesize(p,MIDI_F0).detach()
        # embed
        za = clap.embed_audio(audio).detach()
        # save 
        records.append({'p':p.cpu().numpy(),'za':za.cpu().numpy()})
        

torch.save(records,'artefacts/records.pt')


# %%
