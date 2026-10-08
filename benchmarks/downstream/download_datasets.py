"""Run through download_datasets.sh on a compute node."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import traceback

assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
root = Path('/scratch/mmoslem3/data')
archives = root / '.downstream-downloads'
archives.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'OT-SSL-DD'))
from downstream_data import locate, read_split

def fetch(name, url, checksum=None):
    path = archives / name
    if not path.exists():
        partial = path.with_name(path.name + '.partial')
        print(f'DOWNLOAD {name}', flush=True)
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--retry', '3', '--connect-timeout', '30', '--max-time', '2400',
                        '--speed-limit', '1024', '--speed-time', '120', '--continue-at', '-',
                        '--output', str(partial), url], check=True)
        partial.rename(path)
    if checksum:
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'md5').hexdigest()
        if actual != checksum:
            raise ValueError(f'Checksum mismatch: {path}: {actual}')
    return path

def unpack(path, destination):
    print(f'EXTRACT {path.name}', flush=True)
    with tarfile.open(path) as archive:
        archive.extractall(destination, filter='data')

def stage(dataset):
    try:
        locate(root, dataset)
        return
    except FileNotFoundError:
        pass
    staging = archives / ('staging-' + dataset)
    staging.mkdir(exist_ok=True)
    if dataset == 'CIFAR100':
        unpack(fetch('cifar-100-python.tar.gz', 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                     'eb9058c3a382ffc7106e4002c42a8d85'), staging)
        source = staging / 'cifar-100-python'
    elif dataset == 'CUB2011':
        unpack(fetch('CUB_200_2011.tgz', 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1',
                     '97eceeb196236b17998738112f37df78'), staging)
        source = staging / 'CUB_200_2011'
    elif dataset == 'Aircraft':
        unpack(fetch('fgvc-aircraft-2013b.tar.gz', 'https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'), staging)
        source = staging / 'fgvc-aircraft-2013b'
    elif dataset == 'Dogs':
        source = staging / 'dogs'
        source.mkdir(exist_ok=True)
        for name in ('images.tar', 'lists.tar'):
            unpack(fetch('dogs-' + name, 'https://vision.stanford.edu/aditya86/ImageNetDogs/' + name), source)
    elif dataset == 'Flowers':
        source = staging / 'flowers-102'
        source.mkdir(exist_ok=True)
        base = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/'
        unpack(fetch('102flowers.tgz', base + '102flowers.tgz', '52808999861908f626f3c1f4e79d11fa'), source)
        import shutil
        for name, checksum in [('imagelabels.mat', 'e0620be6f572b9609742df49c70aed4d'),
                               ('setid.mat', 'a5357ecc9cb78c4bef273ce3793fc85c')]:
            shutil.copyfile(fetch(name, base + name, checksum), source / name)
    target = root / source.name
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    source.rename(target)

failed = []
for dataset in ('CIFAR100', 'Flowers', 'CUB2011', 'Dogs', 'Aircraft'):
    try:
        stage(dataset)
        counts = []
        for training in (True, False):
            images, labels, paths = read_split(root, dataset, training)
            counts.append(len(labels))
            del images, labels, paths
        print(f'VERIFIED {dataset}: train={counts[0]} test={counts[1]}', flush=True)
    except Exception:
        failed.append(dataset)
        traceback.print_exc()
print(f'FINISHED failed={failed}', flush=True)
sys.exit(bool(failed))
