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
import glob
# get a microphone input with ipython
from IPython.display import Audio
import librosa

SAMPLE_RATE=48000

# disable gradient calculation
torch.set_grad_enabled(False)


# Run on the GPU if it's available
if torch.cuda.is_available():
    device = torch.device("cuda:3")
else:
    device = torch.device("cpu")

clap_model = laion_clap.CLAP_Module(enable_fusion=True, device=device)
clap_model.load_ckpt() # download the default pretrained checkpoint.


def embed_audio(audio_data):
    audio_embed = clap_model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
    return audio_embed

def embed_text(text_data):
    text_embed = clap_model.get_text_embedding(text_data, use_tensor=True)
    return text_embed

#%%

fps = ["./barks/linus_bark.wav","./barks/nicolas_bark.wav"]

# load audio with librosa
audios = [ librosa.load(fp, sr=SAMPLE_RATE)[0] for fp in fps ]

# pad to 8 seconds
audios = [ np.pad(a, (0, SAMPLE_RATE*8 - len(a)), mode="constant") for a in audios ]

print(audios[0].shape)
# downmix to mono

audios = np.array(audios)

print(audios.shape)

def play(a):
    display.display(display.Audio(a, rate=SAMPLE_RATE))

# %%

prompt = "a bad impression of a dog barking"

z_text = embed_text([prompt, prompt])[:1].cpu().numpy()


#%%

print(audios.shape)

z_audios = embed_audio(torch.tensor(audios).to(torch.float32)).cpu().numpy()



# compute cosine distance
cos_simimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
cos_sim = cos_simimilarity(torch.tensor(z_audios), torch.tensor(z_text)).numpy()

print(fps)
print(cos_sim)



# %%
