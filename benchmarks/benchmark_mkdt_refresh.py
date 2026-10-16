"""Current-protocol MKDT refresh: 10k SSL epochs and 1000/500 probes."""

import benchmark_mkdt_multidataset as benchmark


benchmark.PROTOCOL = 'mkdt_10k_probe1000_500_shared_encoder_v1'
benchmark.SIZES = (1, 2)
benchmark.PROBE_EPOCHS = {1: 1000, 5: 500}


if __name__ == '__main__':
    benchmark.main()
