"""Main-table scheduling only; keep existing MKDT SSL/probe identities intact."""
import interactive_mkdt_2k as mkdt

original_write_tsv = mkdt.write_tsv


def main_targets(source, model):
    return (source,)


def write_main_tsv(path, columns, rows):
    original_write_tsv(path, columns, [row for row in rows if row['target'] == row['source']])


def main_print(*values, **kwargs):
    if values and isinstance(values[0], str) and values[0].startswith('Sizes 1%/2%;'):
        values = ('Sizes 1%/2%; labels 1%/5%; MAIN TABLE ONLY on all three BN networks.',) + values[1:]
    print(*values, **kwargs)


if __name__ == '__main__':
    # Scheduling stays outside the hashed evaluator so saved runs remain reusable.
    mkdt.evaluation_targets = main_targets
    mkdt.write_tsv = write_main_tsv
    mkdt.print = main_print
    raise SystemExit(mkdt.main())
