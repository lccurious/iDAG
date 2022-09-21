# Change log

## [Unreleased]
### Added
- multiprocessing distributed training, but which can only support a single trial rather than all trails. The second trail hangs in `torch.distributed.init_progress_group`.