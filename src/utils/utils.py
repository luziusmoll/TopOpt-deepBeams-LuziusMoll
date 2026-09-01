import numpy as np

# Optimality criteria
""" from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """

def oc(x,volfrac,dc,dv,x_min,vol_check=None,move=0.2):
    # Optimality criteria
    """ from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """

    dc=np.array(dc)
    dv=np.array(dv)
    l1=0
    l2=1e9
    xnew=np.zeros(len(x))
    # Hard iteration cap: the relative test (l2-l1)/(l1+l2) is stuck at 1 while
    # l1 == 0 (constraint never brackets, e.g. a degenerate projection), which
    # otherwise halves l2 until it underflows -> divide-by-zero / overflow in the
    # step below. 90 bisections is ~1e-27 relative, far past convergence.
    for _ in range(90):
        lmid=0.5*(l2+l1)
        ratio = -dc / (dv * lmid)
        np.maximum(ratio, 0.0, out=ratio)   # guard sqrt against tiny FP negatives
        xnew[:]= np.maximum(x_min,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(ratio)))))

        if vol_check is None:
            # area-weighted volume fraction: sum(A_e * x_e) / sum(A_e) <= volfrac
            # (dv carries the per-element areas; falls back to a plain mean if dv is all ones)
            too_much = np.sum(dv * xnew) > volfrac * np.sum(dv)
        else:
            # density/projection filter: measure the true volume fraction of the
            # physical (filtered+projected) field for this candidate design
            # (Andreassen et al. 2011 top88 ft=2; Ferrari & Sigmund 2020 ft=3)
            too_much = vol_check(xnew) > volfrac

        if too_much:
            l1=lmid
        else:
            l2=lmid

        if l1 + l2 == 0:
            return xnew
        if (l2 - l1) / (l1 + l2) < 1e-8:
            break

    return xnew