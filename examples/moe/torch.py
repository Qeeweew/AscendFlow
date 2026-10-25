"""Prepared local-expert GMM on the caller's current Torch NPU stream."""


class GroupedGemm:
    """Load one E/K/N/dtype/layout specialization; capacity remains dynamic.

    The caller owns the TorchRuntime and keeps it alive across calls. The
    kernel consumes int64 cumulative exclusive expert ends (type 0) directly.
    It neither packs weights nor converts routing metadata.
    """

    def __init__(self, runtime, library):
        self.kernel = runtime.kernel(library, "grouped_gemm")
        specs = {spec["name"]: spec for spec in self.kernel.metadata["tensors"]}
        self.experts, self.k, self.n = specs["W"]["shape"]
        self.weight_layout = specs["W"]["layout"]

    def __call__(self, x, weight, group_list, *, out=None):
        """Return [capacity,N]; rows after group_list[-1] are zero.

        x: contiguous ND [capacity,K], FP16/BF16.
        weight: [E,K,N], matching compiled ND or FRACTAL_NZ storage.
        group_list: contiguous device int64 [E], 0 <= ends <= capacity,
                    nondecreasing, with no leading zero element.
        out: optional contiguous ND [capacity,N], disjoint from all inputs.

        Only tensor metadata is inspected on the host; no device synchronization.
        """
        if out is None:
            out = x.new_empty((x.shape[0], self.n))
        self.kernel(X=x, W=weight, GroupList=group_list, Y=out)
        return out
