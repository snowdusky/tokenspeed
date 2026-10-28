# DeepSeek V4.1 attention on Ascend

The Ascend operators use the existing shared operator contracts and LCM fields.

SWA stores E4M3 values with group-32 E8M0 scales; global KV stores E2M1 values
with group-16 E4M3 scales; index K stores E2M1 values with group-32 E8M0 scales.
Each page stores all value rows followed by all scale rows. Field strides are
preserved, including interleaved arena views. Software codecs avoid requiring
native FP4/FP8 arithmetic on the device.

Selected attention reads packed pages directly and uses FP32 online softmax,
with the same BF16 decoded-value boundary as the portable reference. It also
accepts the backend's compact BF16 prefill workspace, which already combines
SWA and global rows even when the backend also passes cache handles. Slot and length tensors
remain live inputs during native ACL graph replay.

Index scoring preserves the portable BF16 dot-product, weighted-product and
shard-sum rounding boundaries. Query tiling never changes scoring semantics.
Ascend gathers packed keys and uses CANN BF16 BMM followed by TopK over FP32
scores. Key and score/product scratch is bounded to about 32 MiB per query
tile, except when one history row itself exceeds that cap. Bounded codec
worker grids support large batches without exceeding the device launch limit.
The portable `score_chunk_size` scan hint is validated at the shared entry;
CANN scores a whole history row and bounds memory by query tiles instead.
The short-history all-selection path compacts already ordered IDs without
sorting, including the full 512-row / 2048-block output geometry.
Full selection computes candidate-block maxima independently of selected rows;
Reindex scores only the supplied candidate blocks.

RoPE uses FP32 tensor arithmetic and writes back to the original dtype. Index
key normalization rounds to BF16 before rotation, and index query weights
round after FP32 scaling. Strided query projections rotate in place; the
scaled weights are fresh contiguous outputs. These
operations and the model's Engram signed-square-root gate are capturable by
native ACL graphs; neither requires GE compilation or host tensor copies.

For bitwise eager/replay validation over HCCL, set `HCCL_DETERMINISTIC=true`
before starting the workers. Default BF16 collective reduction order can vary
between repeated eager calls as well as between eager and captured execution.
Checkpoint expansion and Engram lookup use bounded worker grids so real expert
shards and long prefill hash batches stay within Ascend's launch-grid limit.
