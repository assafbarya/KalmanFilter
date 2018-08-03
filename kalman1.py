# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:09:08 2018

@author: assaf
"""
import numpy as np
from scipy.optimize import minimize_scalar, minimize

SIZE = 10000
Q    = 55.
R    = 42

def minusLogLikelihood( v, z, p0 = 1 ):
    q = v[0]
    R = v[1]
    xBefore = np.zeros( z.shape )
    xAfter  = np.zeros( z.shape )    
    pBefore = np.zeros( z.shape )
    pAfter  = np.zeros( z.shape )
    K       = np.zeros( z.shape )

    pBefore[0] = p0
    
    ## at the first step, we only have update
    i=0
    K[i]      = pBefore[i]/(pBefore[i] + R)
    xAfter[0] = xBefore[i] + K[i]*(z[i]-xBefore[i])
    pAfter[0] = (1-K[i])*pBefore[i]
    
    for i in xrange( 1, z.size ):
        # prediction
        xBefore[i] = xAfter[i-1]
        pBefore[i] = pAfter[i-1] + q
        
        # update
        K[i]      = pBefore[i]/(pBefore[i] + R)
        xAfter[i] = xBefore[i] + K[i]*(z[i]-xBefore[i])
        pAfter[i] = (1-K[i])*pBefore[i]
        
    error = xBefore - z
    pBs = pBefore + R
    hmll =   np.log( pBs ) + error * error / pBs # half minus log likelihood, up to a constant term of log(2pi)
    err = error * error

    return np.mean( hmll)

def main():
    n = np.random.randn( SIZE ) * np.sqrt( Q ) 
    x = np.cumsum( n )
    z = x + np.sqrt( R ) * np.random.randn( SIZE )
    x0 = [ 1, 1 ]
    res = minimize( minusLogLikelihood, x0, args = (z,) )
    print ( res )  
    
    
    
    
    
    
    
    
    
    
main()