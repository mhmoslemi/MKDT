"""Continue the existing 3x CIFAR10 noise states to 10x; retain both original runs."""
import math
from pathlib import Path
import shutil

import torch

import worker as base


def extend_state(saved, run_args):
    target, iteration = run_args['Iteration'], saved['iteration']
    assert 0 < iteration < target and saved['scheduler']['last_epoch'] == iteration
    scheduler = dict(saved['scheduler'], T_max=target)
    eta_min = scheduler['eta_min']
    rates = [eta_min + (lr - eta_min) * .5 * (1 + math.cos(math.pi * iteration / target))
             for lr in scheduler['base_lrs']]
    assert len(rates) == len(saved['optimizer']['param_groups'])
    scheduler['_last_lr'] = rates
    optimizer = dict(saved['optimizer'], param_groups=[dict(group, lr=lr)
                     for group, lr in zip(saved['optimizer']['param_groups'], rates)])
    extended = dict(saved, init='noise10x', complete=False, run_args=run_args,
                    scheduler=scheduler, optimizer=optimizer)
    assert optimizer['state'] is saved['optimizer']['state']
    assert all(extended[key] is value for key, value in saved.items()
               if key not in ('init', 'complete', 'run_args', 'scheduler', 'optimizer'))
    return extended


def self_check(_data):
    parameter = torch.nn.Parameter(torch.tensor([.3]))
    optimizer = torch.optim.Adam([parameter], lr=.0015, eps=1e-12)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=.000015)
    for _ in range(3):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
        scheduler.step()
    original = dict(iteration=3, init='noise3x', complete=True, run_args=dict(Iteration=3),
                    scheduler=scheduler.state_dict(), optimizer=optimizer.state_dict(),
                    data=parameter.detach().clone(), rng=torch.get_rng_state())
    extended = extend_state(original, dict(Iteration=10))
    assert original['complete'] and original['scheduler']['T_max'] == 3
    assert math.isclose(original['optimizer']['param_groups'][0]['lr'], .000015)
    assert extended['optimizer']['state'] is original['optimizer']['state']
    assert extended['data'] is original['data'] and extended['rng'] is original['rng']
    optimizer.load_state_dict(extended['optimizer'])
    scheduler.load_state_dict(extended['scheduler'])
    for iteration in range(4, 11):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
        scheduler.step()
        expected = .000015 + (.0015 - .000015) * .5 * (1 + math.cos(math.pi * iteration / 10))
        assert math.isclose(optimizer.param_groups[0]['lr'], expected, abs_tol=1e-12)
    assert scheduler.last_epoch == 10 and math.isclose(optimizer.param_groups[0]['lr'], .000015)
    print('PASS: unchanged source state, preserved Adam/pixels/RNG, extended cosine phase and final learning rate', flush=True)


def fork_parent(percentage, run, job, common):
    parent_run = base.HERE / 'runs/noise3x' / ('pt%g' % percentage)
    parent = parent_run / 'synthetic' / ('res_IIC_CIFAR10_ResNet18BN_%gpercent_final.pt' % percentage)
    destination = Path(job['synthetic_dir']) / parent.name
    origin_file = run / 'extension_origin.json'
    with common.exclusive_run(run):
        if destination.exists():
            origin = base.read(origin_file)
            assert origin['parent_checkpoint'] == str(parent)
            assert common.file_hash(parent) == origin['parent_sha256']
            assert origin['target_iteration'] == job['generation_args']['Iteration']
            return
        with common.exclusive_run(parent_run):
            parent_job = base.read(parent_run / 'job.json')
            for path, expected in parent_job['code_hashes'].items():
                assert common.file_hash(path) == expected, 'Parent implementation changed: ' + path
            saved = torch.load(parent, map_location='cpu', weights_only=True)
            expected_args = dict(job['generation_args'], Iteration=job['generation_args']['Iteration'] * 3 // 10)
            assert parent_job['generation_args'] == saved['run_args'] == expected_args
            assert saved['init'] == 'noise3x' and 0 < saved['iteration'] <= expected_args['Iteration']
            assert len(saved['loss_history']) == saved['iteration']
            assert saved['initialization'] == base.read(parent_run / 'initialization.json')
            assert base.tensor_hash(saved['initial_data']) == saved['initialization']['initial_images_sha256']
            assert saved['teacher_source'] == 'pretrained' and saved['teacher_seed'] == 0
            assert saved['data'].shape == saved['initial_data'].shape == (int(500 * percentage), 3, 32, 32)
            assert torch.isfinite(saved['data']).all() and saved['scheduler']['T_max'] == expected_args['Iteration']
            for field in ('rng', 'numpy_rng', 'objective_rng', 'data_rng', 'real_order', 'real_position'):
                assert field in saved
            parent_hash = common.file_hash(parent)
            extended = extend_state(saved, job['generation_args'])
            origin = dict(parent_checkpoint=str(parent), parent_sha256=parent_hash,
                          from_iteration=saved['iteration'], parent_target_iteration=expected_args['Iteration'],
                          target_iteration=job['generation_args']['Iteration'],
                          preserved='pixels, initial pixels, Adam moments/steps, histories, sampling and all RNG states',
                          scheduler='cosine phase recalculated at current iteration using the 10x total horizon',
                          parent_lrs=[group['lr'] for group in saved['optimizer']['param_groups']],
                          continuation_lrs=[group['lr'] for group in extended['optimizer']['param_groups']])
            common.atomic_json(run / 'initialization.json', saved['initialization'])
            common.atomic_json(origin_file, origin)
            first_png = 'vis_IIC_CIFAR10_ResNet18BN_%gpercent_iter1.png' % percentage
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent.parent / first_png, destination.parent / first_png)
            common.atomic_save(destination, extended)
            assert common.file_hash(parent) == parent_hash, 'Source checkpoint changed during the fork'
            print('FORKED NOISE CONTINUATION: ' + base.json.dumps(origin), flush=True)


def main():
    runner = base.frozen_module(Path(base.__file__), 'noise10x_continuation', [
        ("choices=('kmeans100', 'iic')", "choices=('noise10x',)"),
        ('job = read(baseline_manifest)',
         "job = read(baseline_manifest)\n"
         "    job['noise_iteration_multiplier'] = 10\n"
         "    job['generation_args']['Iteration'] *= 10\n"
         "    job['code_hashes'][CONTINUATION_DRIVER] = common.file_hash(CONTINUATION_DRIVER)"),
        ("    if args.stage == 'evaluation':",
         "    if args.stage == 'generation':\n"
         "        fork_parent(args.percentage, run, job, common)\n\n"
         "    if args.stage == 'evaluation':")])
    runner.CONTINUATION_DRIVER = str(Path(__file__).resolve())
    runner.fork_parent = fork_parent
    runner.self_check = self_check

    def no_fresh_initialization(*_args, **_kwargs):
        raise RuntimeError('10x must resume the forked 3x state; refusing fresh initialization')

    runner.choose_images = no_fresh_initialization
    runner.main()


if __name__ == '__main__':
    main()
