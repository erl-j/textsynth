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
# activation = lambda x: (torch.sin(x)+1.0)/2.0 #torch.nn.functional.sigmoid(x)#

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
    
trn_text_path = "./data/trn.txt"
val_text_path = "./data/val.txt"

trn_texts = [line.rstrip('\n') for line in open(trn_text_path)]
val_texts = [line.rstrip('\n') for line in open(val_text_path)]

clap = CLAPWrapper()

z_text_trn = clap.embed_text(trn_texts)
#%%

class Z2PatchModel(torch.nn.Module):
    def __init__(self, n_patch_params, n_bins, hidden_size, n_layers, n_heads) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.hidden_size = hidden_size
        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=n_heads,
                layers=1,
                dim_feedforward=hidden_size*4,
                dropout=0.1,
                activation="relu",
            ),
            num_layers=n_layers,
            batch_first=True,
        )
        self.positional_encoding = torch.nn.Parameter(
            torch.randn(n_patch_params+1, self.hidden_size)
        )
        self.embedding = torch.nn.Embedding(n_bins, self.hidden_size)
        self.start_token_embedding = torch.nn.Parameter(torch.randn(1, self.hidden_size))
        self.output_layer = torch.nn.Linear(self.hidden_size, self.resolution)

    def forward(self, z: torch.Tensor, pq:torch.Tensor ) -> torch.Tensor:
        pqz = self.embedding(pq)
        # add start token
        pqz = torch.cat([self.start_token_embedding, pqz], dim=1)
        pqz += self.positional_encoding
        zout = self.decoder(
            tgt=pqz,
            memory=z,
            tgt_is_causal=True,
        )
        logits = self.output_layer(zout)
        return logits
    
    def generate(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        pq = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        for i in range(self.n_bins):
            logits = self.forward(z, pq)
            logits = logits[:, -1, :] / temperature
            pq[:, i] = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), 1).squeeze(-1)
        return pq
    
    def loss(self,target,logits):
        return torch.nn.functional.cross_entropy(logits,target)


clap = CLAPWrapper()


#%%

N_QUANTIZATION_BINS=100

pq = int(torch.rand(dummy_p.shape).to(device) * N_QUANTIZATION_BINS)
records = []
PROMPT = "A bass synth with a long release"
zt = clap.embed_text([PROMPT,PROMPT])[:1]

zt_batch = zt.repeat(BATCH_SIZE,1)
generation=0
best_audio = []

model = Z2PatchModel(n_patch_params=synth.get_number_of_parameters(), n_bins=100, hidden_size=64, n_layers=2, n_heads=2).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
archive = []
for gens in tqdm(range(200)):
    pq = model.generate(zt_batch, temperature=0.1)
    p = float(pq)/N_QUANTIZATION_BINS
    # synthesize 
    audio = synth.synthesize(p,MIDI_F0)
    # peak normalize each sample
    peaks = torch.max(torch.abs(audio),dim=1,keepdim=True)[0]
    audio = audio/peaks
    # embed audio
    za = clap.embed_audio(audio)
    # get fitness by measuring similarity to target
    similarity = torch.nn.functional.cosine_similarity(za,zt)

    for i in range(100):
        logits = model(za,pq)
        loss = model.loss(pq,logits)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss.item())
