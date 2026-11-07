from pathlib import Path
import json
import torch
from PIL import Image, ImageDraw

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
source = root / 'res_IIC_TinyImageNet_ResNet18BN_0.2percent_final.pt'
saved = torch.load(source, map_location='cpu', weights_only=True)
x = saved['data'].detach().float()
assert x.shape == (200, 3, 64, 64) and torch.isfinite(x).all()
mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in saved['normalization'])
pixels = ((x * std + mean).clamp(0, 1) * 255 + .5).byte().permute(0, 2, 3, 1).numpy()
folder = root / 'fig'
folder.mkdir(exist_ok=True)
for i, array in enumerate(pixels):
    Image.fromarray(array).save(folder / f'image_{i:03d}.png')
for start in (0, 100):
    sheet = Image.new('RGB', (1000, 1180), 'white')
    draw = ImageDraw.Draw(sheet)
    for offset in range(100):
        i = start + offset
        left, top = (offset % 10)*100, (offset // 10)*118
        sheet.paste(Image.fromarray(pixels[i]).resize((96, 96)), (left, top))
        draw.text((left + 3, top + 98), f'{i:03d}', fill='black')
    sheet.save(root / f'contact_{start:03d}.png')
archive = torch.load('/scratch/mmoslem3/data/tinyimagenet.pt', map_location='cpu', weights_only=True, mmap=True)
print(json.dumps({'checkpoint_iteration': saved.get('iteration'), 'complete': saved.get('complete'), 'normalization': saved['normalization'], 'archive': {k: list(v.shape) if isinstance(v, torch.Tensor) else str(v)[:1200] for k, v in archive.items()}}))
val = archive['images_val']
Image.fromarray(val[170].permute(1, 2, 0).numpy()).save(root / 'validation_170.png')
print('VAL170_LABEL', archive['labels_val'][170].item())
