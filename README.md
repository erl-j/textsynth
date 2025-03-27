![logo](misc/logo.svg)


Generate synthesizer sounds from text prompts with a simple evolutionary algorithm.

**Audio examples here: https://erl-j.github.io/textsynth/**


Synth: https://github.com/torchsynth/torchsynth

Audio-Text cross modal embedding: https://github.com/LAION-AI/CLAP

##  How it works
Start with randomly initialized synthesizer sounds. Each iteration, the current synthesizer sounds are evaluated on how well they match the text prompt. The best sounds are then combined and mutated to generate new sounds for the next iteration. 200 generations w/ 50 samples takes about ~20s on a 3090 (not tested on CPU).

<!-- Text-audio cosine similarity to prompt over generations -->
<img src="misc/evolution.png" alt="example" width="50%" title="text-audio cosine similarity to prompt over generations" height="auto">

## Future work

- Install guide / requirements.txt
- Diversity preservation.
- Open ended exploration.
- RL
- Neural nets?

## Cite

If you found this project useful please cite:

```BibTex
@software{
  textsynth,
  author       = {Nicolas Jonason},
  title        = {TextSynth: Generate synthesizer sounds from text prompts with a simple evolutionary algorithm},
  month        = october,
  year         = 2023,
}
```
