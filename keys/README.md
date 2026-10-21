Public keys only. Private keys never enter a container or a repo.

- `cora.age.pub`: Cora's age public key. Sealed injection-recovery truth is encrypted to it.
- `reveal.age.pub`: public half of the reveal key. Its private half is an environment secret
  (`COSMOS_REVEAL_KEY`) used to unmask decoys after Stage A verdicts are committed.
