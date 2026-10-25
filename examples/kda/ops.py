"""Shared KDA arithmetic, expressed entirely in the ordinary DSL."""

from ascendflow import language as N


@N.inline
def normalize_qk(q, k, work, partial, statistic, dimension):
    square_q = N.mul(q, q, out=work)
    sum_q = N.reduce_sum(
        square_q,
        axis=0,
        out=statistic[:1],
        scratch=partial[: dimension // 64],
        order="r64_then_native_merge",
    )
    stable_q = N.add(sum_q, 1e-6, out=statistic[:1])
    reciprocal_q = N.rsqrt(stable_q, out=statistic[:1])
    inverse_q = N.load_scalar(reciprocal_q)
    normalized_q = N.mul(q, inverse_q, out=q)
    square_k = N.mul(k, k, out=work)
    sum_k = N.reduce_sum(
        square_k,
        axis=0,
        out=statistic[:1],
        scratch=partial[: dimension // 64],
        order="r64_then_native_merge",
    )
    stable_k = N.add(sum_k, 1e-6, out=statistic[:1])
    reciprocal_k = N.rsqrt(stable_k, out=statistic[:1])
    inverse_k = N.load_scalar(reciprocal_k)
    normalized_k = N.mul(k, inverse_k, out=k)
    return normalized_q, normalized_k


@N.inline
def delta_step(state, q, k, decay, value, beta, product, partial, projected, row_scratch, output):
    decayed = N.mul(state, decay[None, :], out=state)
    projected_products = N.mul(decayed, k[None, :], out=product)
    projection = N.reduce_sum(
        projected_products, axis=1, out=projected, scratch=partial, order="block8_then_packed_merge"
    )
    residual = N.sub(value, projection, out=value)
    residual = N.mul(residual, beta, out=value)
    delta = N.mul(k[None, :], residual[:, None], out=product, scratch=row_scratch)
    updated = N.add(decayed, delta, out=state)
    output_products = N.mul(updated, q[None, :], out=product)
    result = N.reduce_sum(
        output_products, axis=1, out=output, scratch=partial, order="block8_then_packed_merge"
    )
    return updated, result
