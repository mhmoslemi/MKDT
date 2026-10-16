"""Long-run full-data SSL pretraining for the CIFAR main tables."""

import benchmark_full_refresh as implementation


# Keep these checkpoints and evaluations distinct from every earlier full-data
# batch, even when another argument happens to match.
implementation.PROTOCOL = 'full_data_main_simclr1000_barlow1500_v1'


if __name__ == '__main__':
    implementation.main()
