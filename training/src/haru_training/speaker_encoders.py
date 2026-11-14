"""Pinned, local-only speaker inference. CPU is explicit, never a GPU fallback."""
from hashlib import sha256
from pathlib import Path

import numpy as np


def verify_model_files(spec):
    folder = Path(spec['directory'])
    if not spec.get('files'):
        raise ValueError('model file hashes are required')
    for name, expected in spec['files'].items():
        if sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError('model file hash mismatch')
    return folder


def load_vad(spec):
    import torch
    path = Path(spec['path'])
    if sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('VAD hash mismatch')
    model = torch.jit.load(str(path), map_location='cpu').eval()

    class Vad:
        def reset_states(self):
            model.reset_states()

        def __call__(self, frame):
            with torch.inference_mode():
                return float(model(torch.from_numpy(frame), 16000).item())

    return Vad()


def load_cpu_encoder(spec):
    folder = verify_model_files(spec)
    if folder.name != spec['revision']:
        raise ValueError('snapshot folder differs from pinned revision')
    import torch
    torch.set_num_threads(4)
    if spec['encoder'] == 'microsoft/wavlm-base-plus-sv':
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        required = {'config.json', 'preprocessor_config.json', 'pytorch_model.bin'}
        if not required <= spec['files'].keys():
            raise ValueError('WavLM model file hashes incomplete')
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(folder), local_files_only=True)
        model = WavLMForXVector.from_pretrained(str(folder), local_files_only=True,
                                               use_safetensors=False, weights_only=True,
                                               trust_remote_code=False).to('cpu').eval()

        def encode(pcm):
            batch = extractor(pcm, sampling_rate=16000, return_tensors='pt', padding=True)
            with torch.inference_mode():
                return model(**batch).embeddings[0].float().numpy()

    elif spec['encoder'] == 'speechbrain/spkrec-ecapa-voxceleb':
        from speechbrain.lobes.features import Fbank
        from speechbrain.processing.features import InputNormalization
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        if not {'embedding_model.ckpt', 'hyperparams.yaml'} <= spec['files'].keys():
            raise ValueError('ECAPA model file hashes incomplete')
        # Architecture and encode_batch(normalize=False) from the official card.
        # Construct modules directly: no YAML execution, symlinks, or hub fetches.
        features = Fbank(n_mels=80).to('cpu').eval()
        normalize = InputNormalization(norm_type='sentence', std_norm=False).to('cpu').eval()
        model = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072],
                           kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                           attention_channels=128, lin_neurons=192).to('cpu').eval()
        model.load_state_dict(torch.load(folder / 'embedding_model.ckpt', map_location='cpu', weights_only=True), strict=True)

        def encode(pcm):
            signal = torch.from_numpy(np.asarray(pcm, dtype=np.float32)).unsqueeze(0)
            with torch.inference_mode():
                value = normalize(features(signal), torch.ones(1))
                return model(value, torch.ones(1)).reshape(-1).float().numpy()

    else:
        raise ValueError('unsupported speaker encoder')
    return encode
