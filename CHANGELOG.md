# Change log

## [Unreleased]
### Added
- Multiprocessing distributed training, but which can only support a single trial rather than all trails. The second trail hangs in `torch.distributed.init_progress_group`.
- New prototypes updating rules which weighted with each appeared individual sample.
- Manual set different seeds for each process.

### Fixed
- Average evaluation results across all processes.
- Create DDP mode dataset sampler for evaluator.
- Manual shuffle DDP dataset during sampling.
