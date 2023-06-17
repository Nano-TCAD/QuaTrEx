"""
Do symbolic calculations to remove the need for hstack and vstack
"""
from sympy import Symbol
from sympy import Matrix

if __name__ == "__main__":
    vh_d1 = Symbol("vh_d1", commutative=False)
    vh_u1 = Symbol("vh_u1", commutative=False)
    vh_l1 = Symbol("vh_l1", commutative=False)
    pr_d1 = Symbol("pr_d1", commutative=False)
    pr_u1 = Symbol("pr_u1", commutative=False)
    pr_l1 = Symbol("pr_l1", commutative=False)
    px_d1 = Symbol("px_d1", commutative=False)
    px_u1 = Symbol("px_u1", commutative=False)
    px_l1 = Symbol("px_l1", commutative=False)
    zero = Symbol("zero", commutative=False)

    # nbc == 2
    vh_d2 = Matrix([[vh_d1, vh_u1], [vh_l1, vh_d1]])
    vh_u2 = Matrix([[zero, zero], [vh_u1, zero]])
    vh_l2 = Matrix([[zero, vh_l1], [zero, zero]])
    pr_d2 = Matrix([[pr_d1, pr_u1], [pr_l1, pr_d1]])
    pr_u2 = Matrix([[zero, zero], [pr_u1, zero]])
    pr_l2 = Matrix([[zero, pr_l1], [zero, zero]])
    px_d2 = Matrix([[px_d1, px_u1], [px_l1, px_d1]])
    px_u2 = Matrix([[zero, zero], [px_u1, zero]])
    px_l2 = Matrix([[zero, px_l1], [zero, zero]])

    # nbc == 3
    vh_d2 = Matrix([[vh_d1, vh_u1, zero], [vh_l1, vh_d1, vh_u1], [zero, vh_l1, vh_d1]])
    vh_u2 = Matrix([[zero, zero, zero], [zero, zero, zero], [vh_u1, zero, zero]])
    vh_l2 = Matrix([[zero, zero, vh_l1], [zero, zero, zero], [zero, zero, zero]])
    pr_d2 = Matrix([[pr_d1, pr_u1, zero], [pr_l1, pr_d1, pr_u1], [zero, pr_l1, pr_d1]])
    pr_u2 = Matrix([[zero, zero, zero], [zero, zero, zero], [pr_u1, zero, zero]])
    pr_l2 = Matrix([[zero, zero, pr_l1], [zero, zero, zero], [zero, zero, zero]])
    px_d2 = Matrix([[px_d1, px_u1, zero], [px_l1, px_d1, px_u1], [zero, px_l1, px_d1]])
    px_u2 = Matrix([[zero, zero, zero], [zero, zero, zero], [px_u1, zero, zero]])
    px_l2 = Matrix([[zero, zero, px_l1], [zero, zero, zero], [zero, zero, zero]])

    mr_d = vh_l2 @ pr_u2 + vh_d2 @ pr_d2 + vh_u2 @ pr_l2
    mr_u = vh_d2 @ pr_u2 + vh_u2 @ pr_d2
    mr_l = vh_l2 @ pr_d2 + vh_d2 @ pr_l2
    lx_d = (vh_l2 @ px_d2 @ vh_u2 +
            vh_l2 @ px_u2 @ vh_d2 +
            vh_d2 @ px_l2 @ vh_u2 +
            vh_d2 @ px_d2 @ vh_d2 +
            vh_d2 @ px_u2 @ vh_l2 +
            vh_u2 @ px_l2 @ vh_d2 +
            vh_u2 @ px_d2 @ vh_l2)

    lx_u = (vh_l2 @ px_u2 @ vh_u2 +
            vh_d2 @ px_d2 @ vh_u2 +
            vh_d2 @ px_u2 @ vh_d2 +
            vh_u2 @ px_l2 @ vh_u2 +
            vh_u2 @ px_d2 @ vh_d2)
    dmr_lu = vh_l2 @ pr_u2
    dmr_ul = vh_u2 @ pr_l2
    dlx_lu = vh_l2 @ px_d2 @ vh_u2 + vh_l2 @ px_u2 @ vh_d2 + vh_d2 @ px_l2 @ vh_u2
    dlx_ul = vh_u2 @ px_d2 @ vh_l2 + vh_u2 @ px_l2 @ vh_d2 + vh_d2 @ px_u2 @ vh_l2
    print("Hello World!")
