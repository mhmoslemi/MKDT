from pathlib import Path
import torch
from PIL import Image, ImageDraw

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
archive = torch.load('/scratch/mmoslem3/data/tinyimagenet.pt', map_location='cpu', weights_only=True, mmap=True)
for wnid in ('n07646821', 'n01855672'):
    if wnid not in archive['classes']:
        print('Not present:', wnid)
        continue
    label = archive['classes'].index(wnid)
    ids = (archive['labels_val'] == label).nonzero().flatten().tolist()
    sheet = Image.new('RGB', (1000, ((len(ids)+9)//10)*118), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, i in enumerate(ids):
        im = Image.fromarray(archive['images_val'][i].permute(1, 2, 0).numpy())
        left, top = j%10*100, j//10*118
        sheet.paste(im.resize((96, 96)), (left, top))
        draw.text((left+2, top+98), str(i), fill='black')
    sheet.save(root / f'validation_{wnid}.png')
    print(wnid, label, len(ids))
