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
import soundfile as sf

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
    
CLIP_DURATION=1
BATCH_SIZE=50
N_PARENTS=64

N_SAVED_PER_TARGET=16

MUTATION_RATE=0.1
TEMPERATURE=0.1

MIDI_F0=53

config = SynthConfig(batch_size=BATCH_SIZE,sample_rate=SAMPLE_RATE,reproducible=False,buffer_size_seconds=CLIP_DURATION)
synth = SynthWrapper(Voice(config).to(device))
dummy_p=synth.get_parameter_tensor()
activation = lambda x: (torch.sin(x)+1.0)/2.0 #torch.nn.functional.sigmoid(x)#

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
    p += torch.randn(p.shape,device=p.device)*MUTATION_RATE
    return p

def crossover(p1,p2):
    mask = torch.rand(p1.shape,device=p1.device)>0.5
    return p1*mask+p2*(~mask)

prompts=[
"kick drum",
"human scream",
"808 bass",
"piano",
"hihat",
"whistling",
"rain",
"bird",
]
for PROMPT in prompts:
    # random p
    p = torch.rand(dummy_p.shape).to(device)*2*np.pi
    records = []
    zt = embed_text([PROMPT,PROMPT])[:1]

    generation=0
    best_audio = []
    for gens in tqdm(range(200)):
        # synthesize 
        audio = synth.synthesize(activation(p),MIDI_F0)
        # peak normalize each sample
        peaks = torch.max(torch.abs(audio),dim=1,keepdim=True)[0]
        audio = audio/peaks
        # embed audio
        za = embed_audio(audio)

        # TODO: diversity preservation

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

        # save best audio
        best_audio.append(audio[torch.argmax(fitness)].detach().cpu().numpy())
        
        if generation%100==0:

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

    sf.write(f"./results/{PROMPT}_final.wav",audio[torch.argsort(-similarity)].flatten().detach().cpu().numpy(),SAMPLE_RATE)

    df = pd.DataFrame(records)

    # edit together all best audio with 0.25 second each
    audio = np.array(best_audio)[:,:int(SAMPLE_RATE*0.5)].flatten()

    sf.write(f"./results/{PROMPT}_evolution.wav",audio,SAMPLE_RATE)