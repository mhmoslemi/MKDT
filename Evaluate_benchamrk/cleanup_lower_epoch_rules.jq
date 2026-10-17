# Only output locations, never code snapshots, generation inputs or caches.
def output_location:
  test("^Evaluate_benchamrk/([^/]*ssl_checkpoints[^/]*|results|random_eval_results)/.+/config\\.json$") or
  test("^Evaluate_benchamrk/slurm/(random|random_individual|random_eval)/batches/[^/]+/results/.+/config\\.json$") or
  test("^benchmarks/(cifar10|cifar100|tinyimagenet|downstream|cross_downstream|mkdt_images_only|mkdt_multidataset|killarney_mkdt_refresh|vulcan_subset_refresh|vulcan_subset_downstream)/results/batch_[^/]+/.+/config\\.json$");
def protected:
  test("(^|/)(code|synthetic|synthesis|generated|buffers|work|targets|selections|cache|cleanup_trash|cleanup_reports)(/|$)") or
  test("^Evaluate_benchamrk/(random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k|kmeans_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k_6seeds|mkdt_ssl_checkpoints_cifar8k6k_tiny5k4k_6seeds)(/|$)") or
  test("^Evaluate_benchamrk/[^/]*eval_results_current[^/]*(/|$)");
def number_or_null: try tonumber catch null;
def current_budget($method; $source; $size):
  if ($source == "CIFAR10" or $source == "CIFAR100") then
    if $size == 1 then 8000 elif $size == 2 then 6000
    elif $size == 0.2 and $method != "mkdt" then 10000 else null end
  elif $source == "TinyImageNet" then
    if $size == 1 then 5000 elif $size == 2 then 4000
    elif $size == 0.2 and $method != "mkdt" then 6000 else null end
  else null end;

(input_filename | ltrimstr($root + "/")) as $relative |
select(($relative | output_location) and (($relative | protected) | not)) |
(.signature.selection_method // .identity.selection_method // .method //
  (if ((.protocol // "") | startswith("mkdt_")) then "mkdt" else null end)) as $method |
select($method == "random" or $method == "kmeans" or $method == "mkdt") |
(.signature.source // .source // .dataset //
  (if ((.protocol // "") | startswith("cifar10_")) then "CIFAR10"
   elif ((.protocol // "") | startswith("mkdt_cifar10_")) then "CIFAR10" else null end)) as $source |
((.signature.subset // .subset_percentage) | number_or_null) as $size |
((.signature.protocol.ssl.subset_epochs // .ssl_epochs) | number_or_null) as $epochs |
current_budget($method; $source; $size) as $budget |
(if $epochs == null or $epochs <= 0 or ($epochs | floor) != $epochs then "KEEP_UNVERIFIED"
 elif $budget == null then "KEEP_NO_CURRENT_BUDGET"
 elif $epochs < $budget then "REMOVE_LOWER"
 else "KEEP_EQUAL_OR_HIGHER" end) as $decision |
[$decision, ($relative | sub("/config\\.json$"; "")), $method, ($source // "unknown"),
 ($size // "unknown"), ($epochs // "unknown"), ($budget // "none")] | @tsv
