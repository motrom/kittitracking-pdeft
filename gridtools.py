#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
tools for manipulating grids
using opencv's warpAffine
"""
import numpy as np
from cv2 import warpAffine, BORDER_TRANSPARENT, BORDER_CONSTANT
from cv2 import INTER_LINEAR, INTER_CUBIC, WARP_INVERSE_MAP

def updateGridwGrid(priorgrid, msmtgrid, viewedgrid, msmt_confusion_matrix):
    """
    given an occupancy prior and measured occupancy, update posterior of occupancy
    prior and posterior are float matrices
    msmtgrid and viewedgrid are boolean matrices
    if there is a msmt in the grid, it is a positive measurement
    if there is no msmt and the tile was viewed, it is a negative measurement
    if the tile is not viewed it is not updated
    P(x=1|z=1) = P(x=1)P(z=1|x=1)/(P(x=1)P(z=1|x=1) + (1-P(x=1))P(z=1|x=0))
    """
    tnp,fpp = msmt_confusion_matrix[0]
    fnp,tpp = msmt_confusion_matrix[1]
    posterior_seen = priorgrid*tpp/(fpp + priorgrid*(tpp-fpp))
    posterior_notseen = priorgrid*fnp/(tnp + priorgrid*(fnp-tnp))
    posterior = priorgrid.copy()
    posterior[msmtgrid] = posterior_seen[msmtgrid]
    notseen = (msmtgrid==False) & viewedgrid
    posterior[notseen] = posterior_notseen[notseen]
    return posterior

def reOrientGrid(priorgrid, transform, initial_val, gridstep, gridstart, gridlen):
    """
    transform = [[cos,-sin,tx],[sin,cos,ty],[0,0,1]]
    shift an occupancy grid
    Obviously, there is error due to imperfect matching of old and new tiles.
    This function does an approximation by finding the old tiles corresponding to
    points evenly spaced within the tile
    """
    T00, T01, T02, T10, T11, T12 = transform[:2,:3].flat
    movex = ((T00-1)*gridstart[0] + T01*gridstart[1] + T02)/gridstep[0]
    movey = ((T11-1)*gridstart[1] + T10*gridstart[0] + T12)/gridstep[1]
    T = np.array([[T11,T10,movey],[T01,T00,movex]])
    return warpAffine(priorgrid, T, (gridlen[1],gridlen[0]),
                      flags=INTER_LINEAR,#+WARP_INVERSE_MAP,
                      borderMode=BORDER_CONSTANT, borderValue=initial_val)


def mixGrid(grid, mixer, outervalue, tempmat=None):
    """
    perform 2d convolution on a grid to emulate propagation between adjacent tiles
    tiles outside the limit of the grid are set to outervalue
    """
    assert mixer.shape[0]%2 and mixer.shape[1]%2
    pad = np.array(mixer.shape)//2
    if tempmat is None:
        tempmat = np.zeros(grid.shape+pad*2, dtype=grid.dtype)
    else:
        assert all(pad*2+grid.shape == tempmat.shape)
    tempmat[pad[0]:-pad[0], pad[1]:-pad[1]] = grid
    tempmat[:pad[0],:] = outervalue
    tempmat[-pad[0]:,:] = outervalue
    tempmat[:,:pad[1]] = outervalue
    tempmat[:,-pad[1]:] = outervalue
    
    viewshape = (grid.shape[0], grid.shape[1], mixer.shape[0], mixer.shape[1])
    view4conv = np.lib.stride_tricks.as_strided(tempmat, viewshape,
                                                tempmat.strides*2, writeable=False)
    grid[:] = np.einsum(view4conv, [0,1,2,3], mixer, [2,3], [0,1])
    
    
    
""" useful polynomial approximation of normal cdf
    source = John D Cooke blog """
def approxNormalCdf(dev): return 1./(1 + np.exp(-.07056 * dev**3 - 1.5976 * dev))
    
def eigTwoxTwo(varx, vary, covxy):
    # from math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/
    T = (varx+vary)*.5
    D = varx*vary-covxy*covxy
    eigval1 = T + np.sqrt(T*T-D)
    eigval2 = 2*T - eigval1
    eigvecnorm = np.hypot(eigval1-vary, covxy)
    return eigval1, eigval2, (eigval1-vary)/eigvecnorm, covxy/eigvecnorm

""" this is an approximate cdf, assuming independent prob in x and y directions
"""
def mapNormal2Grid(meanx, meany, varx, vary, covxy,
                    gridstart, gridstep, gridlen):
    xposs = np.arange(gridlen[0]+1)*gridstep[0] + gridstart[0]
    cdf = approxNormalCdf((xposs-meanx) / varx**.5)
    totalprobinside = cdf[-1] - cdf[0]
    if totalprobinside < 1e-10:
        # very low likelihood of appearance, just set to uniform
        return np.zeros(gridlen) + 1./gridlen[0]/gridlen[1]
    llx = np.diff(cdf) / totalprobinside
    yposs = np.arange(gridlen[1]+1)*gridstep[1] + gridstart[1]
    cdf = approxNormalCdf((yposs-meany) / vary**.5)
    totalprobinside = cdf[-1] - cdf[0]
    if totalprobinside < 1e-10:
        return np.zeros(gridlen) + 1./gridlen[0]/gridlen[1]
    lly = np.diff(cdf) / totalprobinside
    return np.outer(llx, lly)

""" approximate cdf, accounting for xy correlation
make normal cdf grid, do rotation transform to put on rectified grid
don't do scale transform, b.c. you would need to sum probs for increased scale
note: cv2 transformation matrices have y-axis first
"""
def mapNormal2GridRot(meanx, meany, varx, vary, covxy,
                    gridstart, gridstep, gridlen):
    rectvx, rectvy, rectc, rects = eigTwoxTwo(varx, vary, covxy)
    gridcenter = gridstart + gridlen*.5*gridstep
    rotmeanx = rectc*(meanx-gridcenter[0]) + rects*(meany-gridcenter[1]) + gridcenter[0]
    rotmeany = rectc*(meany-gridcenter[1]) - rects*(meanx-gridcenter[0]) + gridcenter[1]
    ingrid = mapNormal2Grid(rotmeanx, rotmeany, rectvx, rectvy,
                            0, gridstart, gridstep, gridlen)
    midx, midy = gridlen*.5 - .5
    T = np.array(((rectc,  rects, midy-rectc*midy-rects*midx),
                  (-rects, rectc, midx-rectc*midx+rects*midy)))
    outgrid = warpAffine(ingrid, T, (gridlen[1], gridlen[0]),
                         flags=INTER_LINEAR,
                         borderMode=BORDER_CONSTANT, borderValue=0.)
    # bilinear interpolation may alter the sum of values
    # problem for probability distributions
    if np.sum(outgrid) > 1: outgrid /= np.sum(outgrid)
    return outgrid
    
"""
return subgrid with probability of occupancy
and subgrid location
default subsize 0 -- just pick the center cell and return this
if cell outside of grid, returns size-0 subgrid
"""
def mapNormal2Subgrid(normalparams, gridstart, gridstep, gridlen, subsize = 0):
    meanx, meany, varx, vary, covxy = normalparams
    tilex = int(np.floor(meanx/gridstep[0]))-gridstart[0]
    tiley = int(np.floor(meany/gridstep[1]))-gridstart[1]
    tilexmin = max(tilex-subsize, 0)
    tilexmax = min(tilex+subsize+1,gridlen[0])
    tileymin = max(tiley-subsize, 0)
    tileymax = min(tiley+subsize+1,gridlen[1])
    subgridstart = np.array((tilexmin, tileymin))
    subgridlen = np.array((tilexmax-tilexmin, tileymax-tileymin))
    if any(subgridlen <= 0): # size-0 subgrid
        return np.array((0,0)), np.zeros((0,0))
    subgrid = mapNormal2GridRot(meanx, meany, varx, vary, covxy,
                                subgridstart + gridstart, gridstep, subgridlen)
    return subgridstart, subgrid
    
    
    
"""
test reorientation, normal mapping, and mixing
"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ioff()
    
    gridstart = np.array((-12.,-6.))
    gridlen = np.array((8,8))
    gridstep = np.array((3.,3.))
    
    
    meanx = -3.
    meany = 8.
    varx = 4.**2
    vary = 3.**2
    covxy = .2*4*3
    normalmean = np.array((meanx, meany))
    normalvar = np.array(((varx, covxy), (covxy, vary)))
    
    ## make a high-res mesh of the normal distribution
    hresgridx, hresgridy = np.meshgrid(np.linspace(-12., 12, 100),
                                       np.linspace(-6., 18, 100))
    precvals, precvec = np.linalg.eigh(normalvar)
#    rectvarx, rectvary, precvecx, precvecy = eigTwoxTwo(4., 4., -1.)
#    precvals = np.array((rectvarx, rectvary))
#    precvec = np.array(((precvecx, -precvecy),(precvecy, precvecx)))
    precvals = 1/precvals
    ll1 = precvals[0]*(precvec[0,0]*(hresgridx-normalmean[0]) +
                       precvec[1,0]*(hresgridy-normalmean[1]))
    ll2 = precvals[1]*(precvec[0,1]*(hresgridx-normalmean[0]) +
                       precvec[1,1]*(hresgridy-normalmean[1]))
    ll = np.exp(-.5*(ll1*ll1 + ll2*ll2))
    
    ## map the normal distribution to the grid
    outgrid = mapNormal2GridRot(meanx, meany, varx, vary, covxy,
                                gridstart, gridstep, gridlen)
    
    ## compare mapped distribution to correct version
    plt.subplot(121).contour(hresgridx, hresgridy, ll)
    outgridForShow = outgrid.T[::-1]
    plt.subplot(122).imshow(outgridForShow)
    plt.show()
    
#    ## re-orient distribution
    transform = np.array(((1.,0,-4),(0,1,0),(0,0,1)))
    #transform = np.array(((.9798, -.2, -2.), (.2, .9798, 0), (0,0,1)))
    initial_val = .1
    reoriented = reOrientGrid(outgrid, transform, initial_val,
                                gridstep, gridstart, gridlen)
    plt.figure(figsize=(10.,8.))
    plt.subplot(221).imshow(outgrid.T[::-1])
    plt.subplot(224).imshow(reoriented.T[::-1])
    plt.show()