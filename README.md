# coqui /xtts-v2

[![Try a demo on Replicate](https://replicate.com/lucataco/xtts-v2/badge)](https://replicate.com/lucataco/xtts-v2)

This is an implementation of the [coqui /xtts-v2](https://github.com/coqui-ai/tts) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

Run predictions:

    cog predict -i text="Hi there, I'm your new voice clone. Try your best to upload quality audio" -i speaker=@female.wav

## Example:

"Hi there, I'm your new voice clone. Try your best to upload quality audio"

![alt text](output.wav)

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

The model weights are not covered by this license. [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) is released under the [Coqui Public Model License 1.0.0](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt), which allows non-commercial use of the model and its outputs only.
