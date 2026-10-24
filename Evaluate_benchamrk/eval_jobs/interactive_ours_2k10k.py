"""Main-table ours evaluation with SimCLR 2k / Barlow Twins 10k epochs."""
import runpy

import interactive_ours as ours

SSL_EPOCHS = {'simclr': 2000, 'barlowtwins': 10000}
original_evaluate_set = ours.evaluate_set
original_expected_protocol = ours.expected_protocol


def evaluate_set(args, synthetic, metadata, model, percentage, method, datasets, hashes):
    def expected_protocol(selection, source, subset):
        protocol = original_expected_protocol(selection, source, subset)
        protocol['ssl']['subset_epochs'] = SSL_EPOCHS[method]
        assert protocol['evaluation']['epochs'] == 100
        return protocol

    # Override the schedule before the shared evaluator builds its checkpoint identity.
    ours.expected_protocol = expected_protocol
    try:
        return original_evaluate_set(args, synthetic, metadata, model, percentage,
                                     method, datasets, hashes)
    finally:
        ours.expected_protocol = original_expected_protocol


if __name__ == '__main__':
    assert ours.SSL_SEEDS == (0, 1) and ours.PROBE_SEEDS == (0, 1, 2)
    ours.evaluate_set = evaluate_set
    print('All datasets/sizes: SimCLR=2000 SSL epochs; Barlow Twins=10000 SSL epochs.', flush=True)
    runpy.run_module('interactive_ours_main', run_name='__main__')
