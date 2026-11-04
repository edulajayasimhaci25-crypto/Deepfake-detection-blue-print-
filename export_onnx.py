import torch
from server import vision_model, audio_model

print('Exporting Vision Transformer to ONNX...')
dummy_v = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    vision_model,
    dummy_v,
    'vision_model.onnx',
    input_names=['input_frames'],
    output_names=['spatial_score'],
    dynamic_axes={'input_frames': {0: 'batch_size'}}
)

print('Exporting Wav2Vec 2.0 Acoustic Model to ONNX...')
dummy_a = torch.randn(1, 16000 * 3)
torch.onnx.export(
    audio_model,
    dummy_a,
    'audio_model.onnx',
    input_names=['audio_waveform'],
    output_names=['acoustic_score'],
    dynamic_axes={'audio_waveform': {0: 'batch_size', 1: 'sequence_length'}}
)

print('ONNX Export Complete: vision_model.onnx and audio_model.onnx generated successfully.')
