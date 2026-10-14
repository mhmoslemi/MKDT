"""Fresh 600-epoch full-data SSL pretraining and evaluation protocol."""

import benchmark_full_refresh as implementation


# Keep these checkpoints and evaluations incompatible with the earlier
# full-data refresh, even when every other argument happens to match.
implementation.PROTOCOL = 'full_data_all_ssl_600epoch_v1'


if __name__ == '__main__':
    implementation.main()
