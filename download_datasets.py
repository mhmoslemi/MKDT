import hashlib
from pathlib import Path
import subprocess
import sys

root = Path('/scratch/mmoslem3/data')
downloads = {}
downloads['cifar-100-python.tar.gz'] = ('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 'eb9058c3a382ffc7106e4002c42a8d85')
downloads['CUB_200_2011.tgz'] = ('https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1', '97eceeb196236b17998738112f37df78')
downloads['fgvc-aircraft-2013b.tar.gz'] = ('https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz', None)
downloads['dogs-images.tar'] = ('http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar', None)
downloads['dogs-lists.tar'] = ('http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar', None)
downloads['102flowers.tgz'] = ('https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz', '52808999861908f626f3c1f4e79d11fa')
downloads['imagelabels.mat'] = ('https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat', 'e0620be6f572b9609742df49c70aed4d')
downloads['setid.mat'] = ('https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat', 'a5357ecc9cb78c4bef273ce3793fc85c')


def fetch(name, url, checksum):
    path = root / name
    partial = path.with_name(path.name + '.partial')
    if not path.exists():
        print(f'Downloading {name}', flush=True)
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '3', '--connect-timeout', '30', '--max-time', '2400', '--speed-limit', '1024', '--speed-time', '120', '--continue-at', '-', '--output', str(partial), url], check=True)
    candidate = path if path.exists() else partial
    if checksum:
        with candidate.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'md5').hexdigest()
        if actual != checksum:
            raise ValueError(f'Checksum mismatch: {candidate}')
    if candidate == partial:
        partial.rename(path)
    print(f'Saved: {path}', flush=True)


def main():
    root.mkdir(parents=True, exist_ok=True)
    failed = False
    for name, (url, checksum) in downloads.items():
        try:
            fetch(name, url, checksum)
        except Exception as error:
            failed = True
            print(f'{name}: {error}', file=sys.stderr, flush=True)
    return failed


if __name__ == '__main__':
    sys.exit(main())
