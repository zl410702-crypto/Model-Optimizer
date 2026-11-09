"""Run migrated BEVFormer-tiny on the original three-frame FP32 reference."""

import argparse
import hashlib
import json
from pathlib import Path

import modelopt
import numpy as np
import torch

from .inference_only.detection import decode_detections
from .inference_only.model import BEVFormerTiny


REPO = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = (REPO.parent / 'BEVFormer' / 'golden_tensors_fp32'
                     / 'fcbccedd61424f1b85dcbf8f897f9754_3e8750f3')
DEFAULT_CHECKPOINT = Path('/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth')


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compare(actual, expected):
    actual = actual.detach().cpu().numpy()
    if actual.shape != expected.shape:
        raise ValueError('Shape mismatch: {} != {}'.format(actual.shape, expected.shape))
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {'MaxAE': float(np.abs(difference).max()),
            'MAE': float(np.abs(difference).mean()),
            'MSE': float(np.mean(difference ** 2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--report', type=Path,
                        default=REPO / 'outputs/bevformer_tiny_fp32_alignment.json')
    args = parser.parse_args()
    if modelopt.__version__ != '0.41.0':
        raise RuntimeError('Expected ModelOpt 0.41.0, got {}'.format(modelopt.__version__))
    if not 1 <= args.frames <= 3:
        raise ValueError('The reference contains three frames')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for full-size BEVFormer inference')
    settings = json.loads((args.reference / 'capture_settings.json').read_text())
    if settings['cudnn_allow_tf32'] or settings['cuda_matmul_allow_tf32']:
        raise ValueError('Expected the TF32-disabled FP32 reference')
    checkpoint_hash = file_sha256(args.checkpoint)
    if checkpoint_hash != settings['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the reference')

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = BEVFormerTiny().cuda().eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state, strict=True)
    if len(state) != 643:
        raise ValueError('Unexpected checkpoint item count: {}'.format(len(state)))

    features = {}
    model.img_backbone.register_forward_hook(
        lambda module, inputs, output: features.__setitem__('backbone/c5.npy', output[-1]))
    model.img_neck.register_forward_hook(
        lambda module, inputs, output: features.__setitem__('fpn/level_0.npy', output[0]))
    results = {}
    history = None
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
        for index in range(args.frames):
            frame = 'frame_{:03d}'.format(index)
            base = args.reference / frame
            image = np.load(base / 'input/images.npy', allow_pickle=False)
            image_hash = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            if image_hash != settings['input_sha256'][frame]:
                raise ValueError('Image hash differs from the reference: {}'.format(frame))
            images = torch.from_numpy(image[None].copy()).cuda()
            can_bus = torch.from_numpy(np.load(base / 'input/can_bus.npy')).cuda()
            lidar2img = torch.from_numpy(np.load(base / 'input/lidar2img.npy')).cuda()
            shift = torch.from_numpy(np.load(base / 'temporal/shift.npy')).cuda()
            output = model(images, can_bus, lidar2img, images.shape[-2:],
                           prev_bev=history, shift=shift)
            history = output['bev_embed'].detach()
            boxes, scores, labels = decode_detections(
                output['all_cls_scores'], output['all_bbox_preds'])[0]
            tensors = {
                **features,
                'bev/final_bev.npy': output['bev_embed'],
                'detection/cls_scores.npy': output['all_cls_scores'],
                'detection/bbox_preds.npy': output['all_bbox_preds'],
                'detection/decoded_boxes.npy': boxes,
                'detection/decoded_scores.npy': scores,
                'detection/decoded_labels.npy': labels,
            }
            results[frame] = {
                name: compare(tensor, np.load(base / name, allow_pickle=False))
                for name, tensor in tensors.items()
            }
            print(frame, json.dumps(results[frame], sort_keys=True), flush=True)

    report = {
        'modelopt_version': modelopt.__version__, 'torch_version': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'checkpoint_sha256': checkpoint_hash, 'checkpoint_items': len(state),
        'reference': str(args.reference), 'stateful_feedback': True,
        'quantization_enabled': False, 'tf32_enabled': False,
        'frames': results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('Report:', args.report)


if __name__ == '__main__':
    main()
