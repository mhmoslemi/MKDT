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

my $protocol = 'full_data_main_simclr1000_barlow1500_v1';
my @models = qw(ConvNet VGG11 ResNet18);
my @objectives = qw(simclr barlowtwins);
my %ssl_epochs = (simclr => 1_000, barlowtwins => 1_500);
my %probe_epochs = (1 => 500, 5 => 300);

sub result_key {
    return join('|', @_);
}

my @summaries;
find(
    {
        no_chdir => 1,
        wanted   => sub {
            push @summaries, $File::Find::name
                if $File::Find::name =~ m{/evaluations/.*/summary\.json$};
        },
    },
    "$batch/evaluations",
);
die "No full-data summaries found under $batch/evaluations\n" unless @summaries;

my %scores;
for my $path (sort @summaries) {
    open my $handle, '<', $path or die "Cannot read $path: $!\n";
    local $/;
    my $summary = decode_json(<$handle>);
    close $handle;

    my $config = $summary->{config} // {};
    my $source = $config->{source} // '';
    my $target = $config->{target} // '';
    next unless $source eq $target;  # Downstream transfer uses a separate overlay.

    my $model = $config->{model} // '';
    my $objective = $config->{ssl_method} // '';
    my $labels = $config->{label_percentage};
    die "$path has the wrong protocol\n"
        unless ($config->{protocol} // '') eq $protocol;
    die "$path has an unsupported source\n"
        unless $source =~ /^(?:CIFAR10|CIFAR100)$/;
    die "$path has an unsupported model\n"
        unless grep { $_ eq $model } @models;
    die "$path has an unsupported SSL objective\n"
        unless exists $ssl_epochs{$objective};
    die "$path is not a full-data result\n"
        unless ($config->{method} // '') eq 'full'
            && ($config->{subset_percentage} // -1) == 100;
    die "$path has the wrong SSL epoch budget\n"
        unless ($config->{ssl_epochs} // -1) == $ssl_epochs{$objective};
    die "$path has an unsupported label budget\n"
        unless defined $labels && exists $probe_epochs{$labels};
    die "$path has the wrong probe epoch budget\n"
        unless ($config->{probe_epochs} // -1) == $probe_epochs{$labels};
    die "$path is incomplete\n"
        unless $summary->{complete}
            && ($summary->{completed_runs} // 0) == 10
            && ($summary->{expected_runs} // 0) == 10
            && join(',', @{$summary->{seeds} // []}) eq '0,1,2,3,4,5,6,7,8,9';
    die "$path lacks numeric aggregates\n"
        unless defined $summary->{mean_percent} && defined $summary->{std_percent};

    my $key = result_key($source, $labels, $model, $objective);
    die "Duplicate completed result for $key\n" if exists $scores{$key};
    $scores{$key} = sprintf(
        '\\dsacc{%.2f}{%.2f}',
        $summary->{mean_percent},
        $summary->{std_percent},
    );
}
die "No complete same-dataset long-epoch results found\n" unless keys %scores;

sub overlay_full_data {
    my ($text, $source, $used) = @_;
    my @lines = split /\n/, $text, -1;
    my @output;
    my $labels;

    for (my $index = 0; $index < @lines; $index++) {
        my $line = $lines[$index];
        $labels = $1 if $line =~ /\\multirow\{18\}\{\*\}\{([15])\\%\}/;

        if ($line =~ /100\\%\s*&\s*Full Data/ && defined $labels) {
            my @row_lines = ($line);
            while ($row_lines[-1] !~ /\\\\\s*$/ && $index + 1 < @lines) {
                push @row_lines, $lines[++$index];
            }
            die "Unterminated Full Data row for $source labels=$labels\n"
                unless $row_lines[-1] =~ /\\\\\s*$/;

            my $row = join(' ', @row_lines);
            my @cells = ($row =~ /(\\(?:oldacc|dsacc)\{[-+0-9.]+\}\{[-+0-9.]+\})/g);
            die "Unexpected Full Data cell count for $source labels=$labels\n"
                unless @cells == 6;

            my $position = 0;
            for my $model (@models) {
                for my $objective (@objectives) {
                    my $key = result_key($source, $labels, $model, $objective);
                    if (exists $scores{$key}) {
                        $cells[$position] = $scores{$key};
                        $used->{$key}++;
                    }
                    $position++;
                }
            }

            push @output,
                '& 100\% & Full Data',
                "& $cells[0] & $cells[1]",
                "& $cells[2] & $cells[3]",
                "& $cells[4] & $cells[5] \\\\";
            next;
        }
        push @output, $line;
    }
    return join("\n", @output);
}

my %used;
my %table_sources = (
    "$tables/cifar10_main.tex"  => 'CIFAR10',
    "$tables/cifar100_main.tex" => 'CIFAR100',
);
for my $path (sort keys %table_sources) {
    open my $handle, '<', $path or die "Cannot read $path: $!\n";
    local $/;
    my $text = <$handle>;
    close $handle;

    $text = overlay_full_data($text, $table_sources{$path}, \%used);

    my $temporary = "$path.tmp.$$";
    open my $output, '>', $temporary or die "Cannot write $temporary: $!\n";
    print {$output} $text;
    close $output or die "Cannot close $temporary: $!\n";
    rename $temporary, $path or die "Cannot replace $path: $!\n";
}

my @unused = grep { !exists $used{$_} } sort keys %scores;
die "Completed results did not map to Full Data cells:\n", join("\n", @unused), "\n"
    if @unused;
my @multiply_used = grep { $used{$_} != 1 } sort keys %used;
die "Full Data results mapped more than once:\n", join("\n", @multiply_used), "\n"
    if @multiply_used;

print scalar(keys %used),
      " verified long-epoch Full Data cells updated from $batch.\n";
