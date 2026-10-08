import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

root = Path('/scratch/mmoslem3/data')
archives = root / '.downstream-downloads'

folders = {'CIFAR100': 'cifar-100-python', 'Flowers': 'flowers-102', 'CUB2011': 'CUB_200_2011', 'Dogs': 'dogs', 'Aircraft': 'fgvc-aircraft-2013b'}

def fetch(name, url, checksum=None):
    path = archives / name
    if not path.exists():
        partial = path.with_name(path.name + '.partial')
        print(f'DOWNLOAD {name}', flush=True)
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '3', '--connect-timeout', '30', '--max-time', '2400', '--speed-limit', '1024', '--speed-time', '120', '--continue-at', '-', '--output', str(partial), url], check=True)
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
    target = root / folders[dataset]
    if (target / '.download-complete').exists():
        print(f'{dataset}: already downloaded', flush=True)
        return
    if target.exists():
        raise FileExistsError(f'{target} already exists without a completion marker')
    staging = archives / ('staging-' + dataset)
    staging.mkdir(exist_ok=True)
    if dataset == 'CIFAR100':
        unpack(fetch('cifar-100-python.tar.gz', 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 'eb9058c3a382ffc7106e4002c42a8d85'), staging)
        source = staging / 'cifar-100-python'
    elif dataset == 'CUB2011':
        unpack(fetch('CUB_200_2011.tgz', 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1', '97eceeb196236b17998738112f37df78'), staging)
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
        for name, checksum in [('imagelabels.mat', 'e0620be6f572b9609742df49c70aed4d'), ('setid.mat', 'a5357ecc9cb78c4bef273ce3793fc85c')]:
            shutil.copyfile(fetch(name, base + name, checksum), source / name)
    (source / '.download-complete').touch()
    source.rename(target)

def main():
    archives.mkdir(parents=True, exist_ok=True)
    failed = []
    for dataset in folders:
        try:
            stage(dataset)
            print(f'{dataset}: ready at {root / folders[dataset]}', flush=True)
        except Exception as error:
            failed.append(dataset)
            print(f'{dataset}: {error}', file=sys.stderr, flush=True)
    return bool(failed)


if __name__ == '__main__':
    sys.exit(main())
