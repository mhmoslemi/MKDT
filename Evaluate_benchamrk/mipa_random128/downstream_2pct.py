"""Five downstream probes from each existing 2% random-init/batch-128 encoder."""
import sys

import worker as base


def downstream(job, reference):
    common = reference['common']
    source, subset, target, labels = (job[k] for k in ('source', 'subset', 'target', 'labels'))
    assert subset == 2 and target in base.TARGETS[source] and labels in (1, 5)
    folder = base.encoder_folder(source, subset)
    with (folder / '.lock').open('a') as lock:
        base.fcntl.flock(lock, base.fcntl.LOCK_SH | base.fcntl.LOCK_NB)
        config, marker = base.read(folder / 'config.json'), base.read(folder / 'ssl_complete.json')
        signature = config['signature']
        assert common.digest_json(signature) == config['identity']['signature']
        assert signature['generation_args'] == job['generation_args'] and signature['selection'] == job['selection']
        assert signature['teacher'] == job['teacher'] and signature['driver_code_hashes'] == job['code_hashes']
        assert signature['source'] == source and signature['subset'] == subset and signature['ssl'] == 'simclr'
        assert signature['code_hashes'] == common.code_hashes()
        checkpoint = folder / 'encoder.pt'
        assert marker['complete'] and marker['epochs'] == 2000 and marker['identity'] == config['identity']
        assert marker['checkpoint_file_sha256'] == common.file_hash(checkpoint)
        encoder, _, trained = common.load_encoder(checkpoint, config['identity'], 'cpu')
        assert trained['epochs'] == 2000 and trained['encoder_sha256'] == marker['encoder_sha256']
        del encoder, trained
        with common.exclusive_run(folder / 'probe_locks' / ('%s_lbl%d' % (target, labels))):
            base.summarize(job, reference, checkpoint, config, marker, target, labels)


if __name__ == '__main__':
    job = base.read(base.Path(sys.argv[1]).resolve())
    assert job['stage'] == 'downstream' and job['subset'] == 2
    for relative, expected in job['downstream_code_hashes'].items():
        assert base.hashlib.sha256((base.ROOT / relative).read_bytes()).hexdigest() == expected
    # Keep the original worker unchanged: active training jobs pin its file hash.
    base.downstream = downstream
    base.main()
