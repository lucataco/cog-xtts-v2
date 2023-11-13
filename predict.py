# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
from TTS.api import TTS

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.environ["COQUI_TOS_AGREED"] = "1"
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')

    def predict(
        self,
        text: str = Input(
            description="Text to synthesize",
            default="Hi there, I'm your new voice clone. Try your best to upload quality audio"
        ),
        speaker: Path = Input(description="Original speaker audio (wav, mp3, m4a, ogg, or flv)"),
        language: str = Input(
            description="Output language for the synthesised speech",
            choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn"],
            default="en"
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        speaker_wav = "speaker.wav"
        filter = "highpass=75,lowpass=8000"
        # ffmpeg convert to wav and apply afftn denoise filter
        os.system(f"ffmpeg -i {speaker} -af {filter} {speaker_wav}")

        path = self.model.tts_to_file(
            text=text, 
            file_path = "output.wav",
            speaker_wav = speaker_wav,
            language = language
        )

        return Path(path)
