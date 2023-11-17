#%%
import torch
from torchsynth.synth import Voice,SynthConfig
from shared import SynthWrapper,CLAPWrapper
from prompts import prompts
from IPython.display import Audio,display
#%%
BATCH_SIZE=60
SAMPLE_RATE = 48000
CLIP_DURATION=1
MIDI_F0 = 53
device = "cuda:1"
config = SynthConfig(batch_size=BATCH_SIZE,sample_rate=SAMPLE_RATE,reproducible=False,buffer_size_seconds=CLIP_DURATION)
synth = SynthWrapper(Voice(config).to(device))
dummy_p=synth.get_parameter_tensor()
clap = CLAPWrapper(device=device)
records = torch.load('artefacts/records.pt')
p = torch.cat([torch.tensor(r['p']) for r in records])
za = torch.cat([torch.tensor(r['za']) for r in records])

#%%
print(f"Number of prompts: {len(prompts)}")
prompts = [p.replace('.','') for p in prompts]
prompts = list(set(prompts))
# shuffle prompts
import random
random.seed(0)
prompts = random.sample(prompts,len(prompts))
prompts = ["a snare drum","a hihat","a kick drum","a bass drum","a tom drum","a cymbal","a cowbell","a woodblock","a clave","a rimshot"]
# prompts =["A synthesizer sound that sounds like a "+p for p in prompts]
print(f"Number of prompts: {len(prompts)} after filtering duplicates")

zts = []
for i in range(0,len(prompts),BATCH_SIZE):
    print(f"Processing batch {i}")
    zts.append(clap.embed_text(prompts[i:i+BATCH_SIZE]).detach())
zt = torch.cat(zts)
za = za.to(device)


#%%
# remove mean from za
# za-zt cosine similarity
za_norm = za / za.norm(dim=-1,keepdim=True)
zt_norm = zt / zt.norm(dim=-1,keepdim=True)
similarity = zt_norm @ za_norm.T
p = p.to(device)
for text_idx,text in enumerate(prompts[:10]):
    print(f"Text: {text}")
    # print top 5 most similar
    argsort = torch.argsort(similarity[text_idx,:],descending=True)
    bestp = p[argsort[:BATCH_SIZE]].to(device)
    a=synth.synthesize(bestp,MIDI_F0=MIDI_F0)

    # play first 3
    print("Playing first 3")
    display(Audio(a[:3].detach().flatten().cpu().numpy(),rate=SAMPLE_RATE))
    

#%%


# %%
