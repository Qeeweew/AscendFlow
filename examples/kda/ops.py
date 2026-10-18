"""Shared KDA arithmetic, expressed entirely in the ordinary DSL."""

from ascendflow import language as N


@N.inline
def normalize_qk(q, k, work, partial, statistic, dimension):
    N.mul(q, q, out=work)
    N.reduce_sum(
        work,
        axis=0,
        out=statistic[:1],
        scratch=partial[: dimension // 64],
        order="r64_then_native_merge",
    )
    N.add(statistic[:1], 1e-6, out=statistic[:1])
    N.rsqrt(statistic[:1], out=statistic[:1])
    inverse_q = N.load_scalar(statistic[:1])
    N.mul(q, inverse_q, out=q)
    N.mul(k, k, out=work)
    N.reduce_sum(
        work,
        axis=0,
        out=statistic[:1],
        scratch=partial[: dimension // 64],
        order="r64_then_native_merge",
    )
    N.add(statistic[:1], 1e-6, out=statistic[:1])
    N.rsqrt(statistic[:1], out=statistic[:1])
    inverse_k = N.load_scalar(statistic[:1])
    N.mul(k, inverse_k, out=k)


@N.inline
def delta_step(state, q, k, decay, value, beta, product, partial, projected, row_scratch, output):
    N.mul(state, decay[None, :], out=state)
    N.mul(state, k[None, :], out=product)
    N.reduce_sum(product, axis=1, out=projected, scratch=partial, order="block8_then_packed_merge")
    N.sub(value, projected, out=value)
    N.mul(value, beta, out=value)
    N.mul(k[None, :], value[:, None], out=product, scratch=row_scratch)
    N.add(state, product, out=state)
    N.mul(state, q[None, :], out=product)
    N.reduce_sum(product, axis=1, out=output, scratch=partial, order="block8_then_packed_merge")
