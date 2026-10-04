from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import monotonic

from quanllm_harness import HarnessSettings, PluginPolicy, create_harness

CASES = {
    1: r"""Let
A = [[3, 1-i], [1+i, 2]].
1. Verify whether A is Hermitian.
2. Compute all eigenvalues of A.
3. Verify the eigenvalues using both the trace and determinant of A.
Show your reasoning clearly. If you detect any contradiction in your intermediate calculations,
identify the incorrect step and correct it before continuing.""",
    2: r"""Let v=(1,2,-1)^T and u=(1,1,0)^T. Let P be the orthogonal projection onto the
one-dimensional subspace spanned by u.
1. Construct P. 2. Compute Pv. 3. Compute r=||Pv||^2/||v||^2.
4. Verify consistency with the mathematical properties of an orthogonal projection.
If an intermediate result violates a known invariant, locate and correct the erroneous step.""",
    3: r"""Let R(theta)=[[cos(theta),-sin(theta)],[sin(theta),cos(theta)]] and v=(3,4)^T.
1. Verify that R(theta) is orthogonal. 2. Compute ||v||^2.
3. Compute R(theta)v and verify directly that ||R(theta)v||^2=||v||^2.
4. Evaluate the result at theta=pi/2 as an independent check.
If a contradiction appears, locate and correct the first erroneous step before continuing.""",
    4: r"""A random variable X takes values 0,1,2,3 with probabilities proportional to 1,2,3,4.
1. Find the normalization constant and all four probabilities.
2. Compute E[X] and E[X^2]. 3. Compute Var(X).
4. Verify probability normalization and the invariant Var(X)>=0.
If an intermediate result violates an invariant, locate and correct the erroneous step.""",
    5: r"""Let A=[[2,1],[3,2]].
1. Compute det(A). 2. Compute A^{-1}. 3. Verify the inverse by calculating AA^{-1}.
4. Verify det(A^{-1})=1/det(A).
If a contradiction appears, identify and correct the earliest erroneous step before continuing.""",
    6: r"""Let H=[[2,1,i],[1,2,1],[-i,1,2]].
1. Verify that H is Hermitian. 2. Compute all eigenvalues.
3. Verify the spectrum using both trace and determinant.
4. For each eigenvalue, give and explicitly verify at least one corresponding eigenvector by
checking Hv=lambda*v. If a contradiction appears, locate and repair its actual source.""",
    7: r"""In C^3 let u=(1,i,1)^T and v=(2,1-i,i)^T.
1. Construct the orthogonal projector P onto span(u).
2. Verify P is Hermitian and idempotent. 3. Compute Pv.
4. Compute r=||Pv||^2/||v||^2 using the complex inner product.
5. Verify the result independently from |u^dagger v|^2/(||u||^2||v||^2).
If an invariant fails, localize the first incorrect operation, repair it, and continue.""",
    8: r"""Let A=[[1,2,0],[0,1,1],[2,0,1]] and B=[[1,0,1],[1,1,0],[0,2,1]].
1. Compute det(A) and det(B). 2. Compute C=AB. 3. Compute det(C) directly and verify
det(C)=det(A)det(B). 4. Compute C^{-1}. 5. Verify C C^{-1}=I by strict row-by-column
multiplication. If a contradiction appears, first recompute the local verification operation from
its definition before invalidating already verified upstream quantities.""",
    9: r"""Machine A produces 60% of a factory's items with defect rate 2%; machine B produces
40% with defect rate 5%. An item is observed to be defective.
1. Compute P(A|D) and P(B|D), and verify they sum to 1.
For this same defective item, P(+|A,D)=0.90 and P(+|B,D)=0.70.
2. Compute P(A|D,+) and P(B|D,+).
3. Verify the final posterior both by sequential Bayesian updating and by direct joint-weight
normalization. Use exact fractions where possible and correct any detected inconsistency.""",
    10: r"""Let U=(1/sqrt(2))*[[1,i],[i,1]], V=[[1,0],[0,i]], and
psi=(1/sqrt(2))*(1,1)^T.
1. Verify U and V are unitary. 2. Compute phi=VU psi by applying U and then V.
3. Compute W=VU explicitly and use W psi to compute phi again. 4. Verify both methods agree.
5. Verify ||psi||=||phi||=1. 6. Compute det(U), det(V), and det(W), and verify the exact
identity det(W)=det(V)det(U). If a contradiction occurs, preserve repaired and independently
verified values rather than reverting to an invalidated earlier state.""",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("QUANLLM_MODEL", "QuanLLM-v2.0-qm"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for case_id in range(args.start, args.end + 1):
        question = CASES[case_id]
        started = monotonic()
        settings = HarnessSettings.from_api_key_file(
            model=args.model, plugin_policy=PluginPolicy(enabled=("quanllm-qm-teaching",))
        )
        harness = create_harness(settings)
        try:
            result = harness.answer(question)
            payload = {
                "case": case_id,
                "question": question,
                "elapsed_seconds": monotonic() - started,
                "ok": True,
                "result": result.to_dict(),
            }
        except Exception as exc:
            payload = {
                "case": case_id,
                "question": question,
                "elapsed_seconds": monotonic() - started,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        target = args.output_dir / f"case_{case_id:02d}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": case_id,
                    "ok": payload["ok"],
                    "status": payload.get("result", {}).get("status"),
                    "elapsed_seconds": round(payload["elapsed_seconds"], 2),
                    "output": str(target),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
