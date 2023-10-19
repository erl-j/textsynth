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
import ddsp


# Run on the GPU if it's available
if torch.cuda.is_available():
    device = torch.device("cuda:3")
else:
    device = torch.device("cpu")


def play(a):
    display.display(display.Audio(a, rate=SAMPLE_RATE))


# create torch model
class SynthModel(torch.nn.Module):

    def __init__(self,batch_size, n_harmonics, n_noise_bands, ir_duration, sample_rate, ft_frame_rate, sample_duration, f0, device):
        super().__init__()

        self.sample_rate = sample_rate
        self.harm_synth = ddsp.HarmonicSynth(SAMPLE_RATE)
        self.noise_synth = ddsp.FilteredNoiseSynth(SAMPLE_RATE)
        self.reverb = ddsp.Reverb(SAMPLE_RATE,ir_duration)

        n_ft_frames = int(sample_duration*ft_frame_rate)
        self.n_samples = int(sample_duration*SAMPLE_RATE)

        self.harm_amps_w = torch.nn.Parameter(torch.ones(batch_size,  n_ft_frames, n_harmonics))
        self.noise_amps_w = torch.nn.Parameter(torch.ones(batch_size, n_ft_frames, n_noise_bands))
        self.global_amp_w = torch.nn.Parameter(torch.ones(batch_size,  n_ft_frames,1))
        self.f0 = torch.ones(batch_size,  n_ft_frames,1)*f0
    def forward(self):
        # softmax across harmonics
        self.harm_amps = torch.nn.functional.softmax(self.harm_amps_w,dim=-1)
        # softplus for noise
        self.noise_amps = torch.nn.functional.softplus(self.noise_amps_w)
        # softplus for global amp
        self.global_amp = torch.nn.functional.softplus(self.global_amp_w)
        # relu for 
        harm =  self.harm_synth(self.f0, self.harm_amps, self.global_amp, self.n_samples)
        noise = self.noise_synth(self.noise_amps, self.n_samples)
        audio = self.reverb(harm)
        return audio
        
#%%
SAMPLE_RATE=48000

model = SynthModel(batch_size=1, n_harmonics=100, n_noise_bands=100, ir_duration=1, sample_rate=SAMPLE_RATE, ft_frame_rate=5, sample_duration=2, f0=200, device=device)



    
CLIP_DURATION=1
BATCH_SIZE=50


clap_model = laion_clap.CLAP_Module(enable_fusion=True, device=device)
clap_model.load_ckpt() # download the default pretrained checkpoint.

def embed_audio(audio_data):
    audio_embed = clap_model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
    return audio_embed

def embed_text(text_data):
    text_embed = clap_model.get_text_embedding(text_data, use_tensor=True)
    return text_embed

prompts=[
"a single piano note",
]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for PROMPT in prompts:
    records = []
    zt = embed_text([PROMPT,PROMPT])[:1].detach()
    for gens in tqdm(range(200)):
        # synthesize 
        audio = model()
        # peak normalize each sample
        peaks = torch.max(torch.abs(audio),dim=1,keepdim=True)[0]
        audio = audio/peaks
        # embed audio
        za = embed_audio(audio)

        # get fitness by measuring similarity to target
        similarity = torch.nn.functional.cosine_similarity(za,zt)

        loss = -similarity.mean()

        print(loss.item())

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print sum of harm amps
        print(torch.sum(model.harm_amps_w))

        # heatmap of harmonic w
        sns.heatmap(model.harm_amps.detach().cpu().numpy()[0])
        plt.show()


        play(audio[0].detach().cpu().numpy())






        # for b in range(BATCH_SIZE):
        #     records.append({"generation":gens,"similarity":similarity[b].item()})
        
    #     if gens%100==0:

    #         # clear output
    #         clear_output(wait=True)
    #         # plot sorted fitness
    #         plt.plot(torch.sort(fitness).values.detach().cpu().numpy())
    #         plt.show()

    #         # show scatter plot of similarity
    #         sns.scatterplot(data=pd.DataFrame(records),x="generation",y="similarity",alpha=0.5)
    #         plt.show()
    #         # play audio of samples sorted by similarity
    #         play(audio[torch.argsort(-similarity)].flatten().detach().cpu().numpy())

    # sf.write(f"./results/{PROMPT}_final.wav",audio[torch.argsort(-similarity)].flatten().detach().cpu().numpy(),SAMPLE_RATE)

    # df = pd.DataFrame(records)

    # # edit together all best audio with 0.25 second each
    # audio = np.array(best_audio)[:,:int(SAMPLE_RATE*0.5)].flatten()

    # sf.write(f"./results/{PROMPT}_evolution.wav",audio,SAMPLE_RATE)
# %%
