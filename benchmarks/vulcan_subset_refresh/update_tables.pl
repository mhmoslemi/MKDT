#!/usr/bin/env perl

use strict;
use warnings;
use File::Find qw(find);
use Getopt::Long qw(GetOptions);
use JSON::PP qw(decode_json);

my $batch;
my $tables = 'CVPR_tables';
GetOptions(
    'batch=s'  => \$batch,
    'tables=s' => \$tables,
) or die "Usage: $0 --batch BATCH [--tables CVPR_tables]\n";
die "Usage: $0 --batch BATCH [--tables CVPR_tables]\n" unless defined $batch;

my %ssl_epochs = (
    '0.2' => 20_000,
    '1'   => 10_000,
    '2'   => 10_000,
    '5'   => 8_000,
);
my %probe_epochs = (1 => 1_000, 5 => 500);
my @models = qw(ConvNet VGG11 ResNet18);
my @objectives = qw(simclr barlowtwins);

sub percentage_key {
    my ($value) = @_;
    return '0.2' if abs($value - 0.2) < 1e-9;
    return sprintf('%d', $value);
}

sub result_key {
    return join('|', @_);
}

my @summaries;
find(
    {
        no_chdir => 1,
        wanted   => sub {
            push @summaries, $File::Find::name
                if $File::Find::name =~ m{/runs/.*/summary\.json$};
        },
    },
    "$batch/runs",
);

my %scores;
for my $path (sort @summaries) {
    open my $handle, '<', $path or die "Cannot read $path: $!\n";
    local $/;
    my $raw = <$handle>;
    close $handle;
    my $summary = eval { decode_json($raw) };
    if (!$summary) {
        warn "Skipping summary being written or malformed: $path\n";
        next;
    }
    next unless $summary->{complete};
    next unless ($summary->{completed_runs} // 0) == 10;
    next unless ($summary->{expected_runs} // 0) == 10;
    next unless join(',', @{$summary->{seeds} // []}) eq '0,1,2,3,4,5,6,7,8,9';

    my $config = $summary->{config} // {};
    my $dataset = $config->{target_dataset} // '';
    my $source = $config->{source_dataset} // '';
    my $method = $config->{method} // '';
    my $model = $config->{model} // '';
    my $objective = $config->{ssl_method} // '';
    my $labels = $config->{label_percentage};
    my $subset = percentage_key($config->{subset_percentage} // -1);

    next unless $dataset =~ /^(?:CIFAR10|CIFAR100)$/;
    next unless $source eq $dataset;
    next unless $method =~ /^(?:random|kmeans)$/;
    next unless grep { $_ eq $model } @models;
    next unless grep { $_ eq $objective } @objectives;
    next unless defined $labels && exists $probe_epochs{$labels};
    next unless exists $ssl_epochs{$subset};
    next unless ($config->{probe_epochs} // -1) == $probe_epochs{$labels};
    next unless ($config->{ssl_epochs} // -1) == $ssl_epochs{$subset};
    die "$path lacks numeric aggregates\n"
        unless defined $summary->{mean_percent} && defined $summary->{std_percent};

    my $key = result_key($dataset, $labels, $subset, $method, $model, $objective);
    die "Duplicate completed result for $key\n" if exists $scores{$key};
    $scores{$key} = sprintf(
        '\\dsacc{%.2f}{%.2f}',
        $summary->{mean_percent},
        $summary->{std_percent},
    );
}

sub mark_existing_results_old {
    my ($text) = @_;
    $text =~ s{\\textcolor\{red\}\{\\dsacc\{([-+0-9.]+)\}\{([-+0-9.]+)\}\}}
              {\\oldacc{$1}{$2}}gx;
    $text =~ s{\\dsacc\{([-+0-9.]+)\}\{([-+0-9.]+)\}}
              {\\oldacc{$1}{$2}}g;
    $text =~ s/\\textsuperscript\{\*\}//g;
    return $text;
}

sub add_provenance_legend {
    my ($text) = @_;
    if ($text =~ /Empty entries indicate unavailable results\./) {
        $text =~ s{Empty entries indicate unavailable results\.(?! Values in red use superseded epoch budgets)}
                  {Empty entries indicate unavailable results. Values in red use superseded epoch budgets and are provisional.}g;
    } else {
        $text =~ s{a random encoder\.(?! Values in red use superseded epoch budgets)}
                  {a random encoder. Values in red use superseded epoch budgets and are provisional.}g;
    }
    return $text;
}

sub ensure_old_macro {
    my ($text) = @_;
    $text =~ s{(\\def\\dsacc#1#2\{\$#1\$\{\\footnotesize\$\\pm #2\$\}\}\n)(?!\\def\\oldacc)}
              {$1\\def\\oldacc#1#2{\\textcolor{red}{\\dsacc{#1}{#2}}}\n}g;
    my $score_macros = () = $text =~ /\\def\\dsacc/g;
    my $old_macros = () = $text =~ /\\def\\oldacc/g;
    die "Every table must define old-result formatting ($score_macros dsacc, $old_macros oldacc)\n"
        unless $score_macros == $old_macros;
    return $text;
}

sub overlay_main_results {
    my ($text, $dataset, $used) = @_;
    my @lines = split /\n/, $text, -1;
    my @output;
    my ($labels, $subset);

    for (my $index = 0; $index < @lines; $index++) {
        my $line = $lines[$index];
        $labels = $1 if $line =~ /\\multirow\{18\}\{\*\}\{([15])\\%\}/;
        $subset = percentage_key($1)
            if $line =~ /\\multirow\{4\}\{\*\}\{(0\.2|1|2|5)\\%\}/;

        if ($line =~ /(Random Subset|K-means Subset)/ && defined $labels && defined $subset) {
            my $title = $1;
            my $method = $title eq 'Random Subset' ? 'random' : 'kmeans';
            my @row_lines = ($line);
            while ($row_lines[-1] !~ /\\\\\s*$/ && $index + 1 < @lines) {
                push @row_lines, $lines[++$index];
            }
            my $row = join(' ', @row_lines);
            my @cells = ($row =~ /(\\(?:oldacc|dsacc)\{[-+0-9.]+\}\{[-+0-9.]+\})/g);
            @cells = ('') x 6 unless @cells;
            die "Unexpected cell count for $dataset labels=$labels subset=$subset $method\n"
                unless @cells == 6;

            my $position = 0;
            for my $model (@models) {
                for my $objective (@objectives) {
                    my $key = result_key(
                        $dataset, $labels, $subset, $method, $model, $objective);
                    if (exists $scores{$key}) {
                        $cells[$position] = $scores{$key};
                        $used->{$key}++;
                    }
                    $position++;
                }
            }

            if (grep { length } @cells) {
                my $prefix = $title eq 'Random Subset'
                    ? "& \\multirow{4}{*}{$subset\\%} & Random Subset"
                    : '& & K-means Subset';
                push @output,
                    $prefix,
                    "& $cells[0] & $cells[1]",
                    "& $cells[2] & $cells[3]",
                    "& $cells[4] & $cells[5] \\\\";
            } else {
                push @output, @row_lines;
            }
            next;
        }
        push @output, $line;
    }
    return join("\n", @output);
}

my %used;
my @table_paths = sort glob "$tables/*.tex";
die "No TeX tables found under $tables\n" unless @table_paths;
for my $path (@table_paths) {
    open my $handle, '<', $path or die "Cannot read $path: $!\n";
    local $/;
    my $text = <$handle>;
    close $handle;

    $text = mark_existing_results_old($text);
    $text = ensure_old_macro($text);
    $text = add_provenance_legend($text);
    $text = overlay_main_results($text, 'CIFAR10', \%used)
        if $path =~ /cifar10_main\.tex$/;
    $text = overlay_main_results($text, 'CIFAR100', \%used)
        if $path =~ /cifar100_main\.tex$/;

    my $temporary = "$path.tmp.$$";
    open my $output, '>', $temporary or die "Cannot write $temporary: $!\n";
    print {$output} $text;
    close $output or die "Cannot close $temporary: $!\n";
    rename $temporary, $path or die "Cannot replace $path: $!\n";
}

my @unused = grep { !exists $used{$_} } sort keys %scores;
die "Completed results did not map to table cells:\n", join("\n", @unused), "\n"
    if @unused;
my @multiply_used = grep { $used{$_} != 1 } sort keys %used;
die "Results mapped more than once:\n", join("\n", @multiply_used), "\n"
    if @multiply_used;

print "Marked existing results provisional in ", scalar(@table_paths), " tables; ",
      scalar(keys %used), " current cells overlaid from $batch.\n";
