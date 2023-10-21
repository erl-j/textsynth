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
import einops

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
    

class Z2PatchModel(torch.nn.Module):
    def __init__(self, n_patch_params, n_bins, hidden_size, n_layers, n_heads, z_size) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.hidden_size = hidden_size
        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size*4,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.n_patch_params = n_patch_params
        self.seq_len = n_patch_params+1
        self.positional_encoding = torch.nn.Parameter(
            torch.randn(self.seq_len, self.hidden_size)
        )
        self.embedding = torch.nn.Embedding(n_bins, self.hidden_size)
        self.start_token_embedding = torch.nn.Parameter(torch.randn(1, self.hidden_size))
        self.output_layer = torch.nn.Linear(self.hidden_size, self.n_bins)
        self.tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(self.seq_len)
        self.z_proj = torch.nn.Linear(z_size, self.hidden_size)

    def forward(self, z: torch.Tensor, pq:torch.Tensor ) -> torch.Tensor:
        '''
        z: latent vector (batch, z_dim)
        pq: patch parameter quantization bin idx (batch, n_parameters)
        '''
        assert z.shape[0] == pq.shape[0], "batch size of z and pq must be the same"

        pqz = self.embedding(pq)
        # add start token
        se = self.start_token_embedding.repeat(pqz.shape[0],1,1)
        pqz = torch.cat([pqz,se],dim=1)

        pqz += self.positional_encoding[ :, :]

        zp = self.z_proj(z[:,None,:])

        zout = self.decoder(
            tgt=pqz,
            memory=zp,
            tgt_is_causal=True,
            tgt_mask=self.tgt_mask.to(z.device),
        )
        logits = self.output_layer(zout)
        return logits[:, :-1, :]
    
    def generate(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        # temporary eval
        batch_size = z.shape[0]
        pq = torch.zeros(batch_size, self.n_patch_params).long().to(device)
        with torch.no_grad():
            for i in range(self.n_patch_params):
                logits = self.forward(z, pq=pq)
                logits = logits[:, -1, :] / temperature
                pq[:, i] = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), 1).squeeze(-1)
        return pq
        
    def loss(self,aligned_target,aligned_logits):
        '''
        aligned_target (idxs): (batch, seq_len)
        aligned_logits (logits): (batch, seq_len, n_bins)
        '''
        assert aligned_target.shape[0] == aligned_logits.shape[0], "batch size of target and logits must be the same"
        assert aligned_target.shape[1] == aligned_logits.shape[1], "sequence length of target and logits must be the same"
        # reshape for cross entropy
        aligned_target = einops.rearrange(aligned_target,'b s -> (b s)')
        aligned_logits = einops.rearrange(aligned_logits,'b s n -> (b s) n')
        # compute loss
        loss = torch.nn.functional.cross_entropy(aligned_logits,aligned_target)
        return loss
        
        
    
CLIP_DURATION=1
BATCH_SIZE=60
TEMPERATURE=0.1
MIDI_F0=53
CLAP_Z_SIZE = 512

config = SynthConfig(batch_size=BATCH_SIZE,sample_rate=SAMPLE_RATE,reproducible=False,buffer_size_seconds=CLIP_DURATION)
synth = SynthWrapper(Voice(config).to(device))
dummy_p=synth.get_parameter_tensor()
clap = CLAPWrapper()
N_QUANTIZATION_BINS=100

pq = (torch.rand(dummy_p.shape).to(device) * N_QUANTIZATION_BINS).long()
PROMPT = "A bass synth"
zt = clap.embed_text([PROMPT,PROMPT])[:1]

zt_batch = zt.repeat(BATCH_SIZE,1)
best_audio = []


#%%
model = Z2PatchModel(n_patch_params=dummy_p.shape[-1], n_bins=100, hidden_size=32, n_layers=2, n_heads=2, z_size=CLAP_Z_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
archive = []
records = []

for gen in tqdm(range(10000)):
    with torch.no_grad():
        pq = model.generate(zt_batch, temperature=1.0).detach()
        # heatmap of pq
        p = (pq.float())/N_QUANTIZATION_BINS
        # print(min and max of each parameter)
        # synthesize 
        audio = synth.synthesize(p+0.01,MIDI_F0).detach()
        
        # # peak normalize each sample
        peaks = torch.max(torch.abs(audio),dim=1,keepdim=True)[0]
        audio = audio/peaks
        # embed audio
        za = clap.embed_audio(audio).detach()
        # get fitness by measuring similarity to target
        similarity = torch.nn.functional.cosine_similarity(za,zt).detach()

        for b in range(BATCH_SIZE):
            records.append({
                "generation":gen,
                "similarity":similarity[b].item(),
                "pq":pq[b],
            })
        

    if gen % 10 == 0:
        # clear figure
        clear_output(wait=True)
        

        sns.heatmap(pq.cpu().numpy())
        plt.show()

        # plot audio
        plt.plot(audio.flatten().cpu().numpy())
        plt.show()

        # plot similarity over time
        df = pd.DataFrame(records)

        print(df.head())
        # scatter plot
        sns.scatterplot(data=df,x="generation",y="similarity")
        plt.show()



        
        # play(audio.flatten().cpu().numpy())


    # print mean similarity
    print(f"mean similarity: {similarity.mean().item()}")

    for i in range(1):
        logits = model(za,pq)
        loss = model.loss(pq,logits)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(f"loss: {loss.item()}")
# %%
