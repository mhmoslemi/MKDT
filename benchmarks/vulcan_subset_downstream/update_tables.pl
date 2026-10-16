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
my @objectives = qw(simclr barlowtwins);
my %targets = (
    CIFAR10  => [qw(CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet)],
    CIFAR100 => [qw(TinyImageNet CIFAR10 Aircraft CUB2011 Dogs Flowers)],
);
my %models = (
    CIFAR10  => [qw(ConvNet VGG11 ResNet18)],
    CIFAR100 => [qw(ResNet18)],
);

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
die "No downstream summaries found under $batch/runs\n" unless @summaries;

my %scores;
for my $path (sort @summaries) {
    open my $handle, '<', $path or die "Cannot read $path: $!\n";
    local $/;
    my $summary = decode_json(<$handle>);
    close $handle;

    die "$path is incomplete\n" unless $summary->{complete};
    die "$path does not contain 10 completed runs\n"
        unless ($summary->{completed_runs} // 0) == 10
            && ($summary->{expected_runs} // 0) == 10;
    die "$path has unexpected seeds\n"
        unless join(',', @{$summary->{seeds} // []}) eq '0,1,2,3,4,5,6,7,8,9';

    my $config = $summary->{config} // {};
    my $source = $config->{source_dataset} // '';
    my $target = $config->{target_dataset} // '';
    my $method = $config->{method} // '';
    my $model = $config->{model} // '';
    my $objective = $config->{ssl_method} // '';
    my $labels = $config->{label_percentage};
    my $subset = percentage_key($config->{subset_percentage} // -1);

    die "$path has unsupported source $source\n" unless exists $targets{$source};
    die "$path has unsupported target $target for $source\n"
        unless grep { $_ eq $target } @{$targets{$source}};
    die "$path has unsupported method $method\n"
        unless $method =~ /^(?:random|kmeans)$/;
    die "$path has unsupported model $model for $source\n"
        unless grep { $_ eq $model } @{$models{$source}};
    die "$path has unsupported SSL objective $objective\n"
        unless grep { $_ eq $objective } @objectives;
    die "$path has unsupported label percentage\n"
        unless defined $labels && exists $probe_epochs{$labels};
    die "$path has unsupported subset percentage $subset\n"
        unless exists $ssl_epochs{$subset};
    die "$path has wrong linear-probe protocol\n"
        unless ($config->{probe_protocol} // '') eq 'frozen_encoder';
    die "$path has wrong probe epoch budget\n"
        unless ($config->{probe_epochs} // -1) == $probe_epochs{$labels};
    die "$path has wrong SSL epoch budget\n"
        unless ($config->{ssl_epochs} // -1) == $ssl_epochs{$subset};
    die "$path lacks numeric aggregates\n"
        unless defined $summary->{mean_percent} && defined $summary->{std_percent};

    my $key = result_key(
        $source, $model, $labels, $subset, $method, $target, $objective);
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

sub overlay_downstream_results {
    my ($text, $source, $used) = @_;
    my @lines = split /\n/, $text, -1;
    my @output;
    my ($model, $labels, $subset);

    for (my $index = 0; $index < @lines; $index++) {
        my $line = $lines[$index];

        if ($line =~ /\\label\{tab:downstream-(convnet|vgg11|resnet18)\}/) {
            my %names = (
                convnet => 'ConvNet',
                vgg11 => 'VGG11',
                resnet18 => 'ResNet18',
            );
            $model = $names{$1};
        } elsif ($line =~ /\\label\{tab:cifar100-downstream-resnet18\}/) {
            $model = 'ResNet18';
        }
        $labels = $1 if $line =~ /\\multirow\{18\}\{\*\}\{([15])\\%\}/;
        $subset = percentage_key($1)
            if $line =~ /\\multirow\{4\}\{\*\}\{(0\.2|1|2|5)\\%\}/;

        if ($line =~ /(Random Subset|K-means Subset)/
                && defined $model && defined $labels && defined $subset) {
            my $title = $1;
            my $method = $title eq 'Random Subset' ? 'random' : 'kmeans';
            my @row_lines = ($line);
            while ($row_lines[-1] !~ /\\\\\s*$/ && $index + 1 < @lines) {
                push @row_lines, $lines[++$index];
            }
            die "Unterminated table row for $source $model labels=$labels PT=$subset $method\n"
                unless $row_lines[-1] =~ /\\\\\s*$/;

            my $row = join(' ', @row_lines);
            my @cells = ($row =~ /(\\(?:oldacc|dsacc)\{[-+0-9.]+\}\{[-+0-9.]+\})/g);
            @cells = ('') x 12 unless @cells;
            die "Unexpected cell count for $source $model labels=$labels PT=$subset $method\n"
                unless @cells == 12;

            my $position = 0;
            my $changed = 0;
            for my $target (@{$targets{$source}}) {
                for my $objective (@objectives) {
                    my $key = result_key(
                        $source, $model, $labels, $subset, $method,
                        $target, $objective);
                    if (exists $scores{$key}) {
                        $cells[$position] = $scores{$key};
                        $used->{$key}++;
                        $changed++;
                    }
                    $position++;
                }
            }

            if ($changed) {
                my $prefix = $title eq 'Random Subset'
                    ? "& \\multirow{4}{*}{$subset\\%} & Random Subset"
                    : '& & K-means Subset';
                push @output, $prefix;
                for my $pair (0 .. 5) {
                    my $left = 2 * $pair;
                    push @output, "& $cells[$left] & $cells[$left + 1]"
                        . ($pair == 5 ? ' \\\\' : '');
                }
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
my %table_sources = (
    "$tables/cifar10_downstream.tex"  => 'CIFAR10',
    "$tables/cifar100_downstream.tex" => 'CIFAR100',
);
for my $path (sort keys %table_sources) {
    open my $handle, '<', $path or die "Cannot read $path: $!\n";
    local $/;
    my $text = <$handle>;
    close $handle;

    $text = mark_existing_results_old($text);
    $text = overlay_downstream_results($text, $table_sources{$path}, \%used);

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

print scalar(keys %used),
      " downstream cells updated from complete 10-seed results in $batch.\n";
