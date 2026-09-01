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
    # reshape to perform vector operations
    xnew=np.zeros(len(x))
    while (l2-l1)/(l1+l2)>1e-8:
        lmid=0.5*(l2+l1)
        xnew[:]= np.maximum(x_min,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/(dv*lmid))))))

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

        # with out this float division by 0 can occour in the while loop criteria (additional line compared to sigmund 200 line implementation)
        if l1 + l2 == 0:
            return xnew

    return xnew