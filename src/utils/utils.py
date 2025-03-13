import numpy as np

# Optimality criteria
""" from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """

def oc(x,volfrac,dc,dv,x_min):
    # Optimality criteria
    """ from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """

    dc=np.array(dc)
    l1=0
    l2=1e9
    move=0.2
    # reshape to perform vector operations
    xnew=np.zeros(len(x))
    while (l2-l1)/(l1+l2)>1e-8:
        lmid=0.5*(l2+l1)
        xnew[:]= np.maximum(x_min,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
        
        # if np.mean(dv*xnew)> np.mean(dv*volfrac):
        if np.mean(xnew)> volfrac:   # this assumes that all elements have a comparable area. If that is not the case, a scaling with the element areas is necessary
            l1=lmid
        else:
            l2=lmid
            
        # with out this float division by 0 can occour in the while loop criteria (additional line compared to sigmund 200 line implementation)
        if l1 + l2 == 0:
            return xnew
        
    return xnew