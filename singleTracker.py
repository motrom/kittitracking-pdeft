# -*- coding: utf-8 -*-
"""
last mod 6/3/19
sample form:
    x center, y center, angle, len, wid, speed, ang vel, flattened 7x7 cov of all
prepped sample:
    mean state w/o speed
    cov state w/o speed
    (H mtx implicitly [I 0])
msmt form:
    mean and cov of observables [x,y,a,l,w]
prepped msmt:
    mean and cov of observables [x,y,a,l,w]
"""

import numpy as np
from math import hypot, atan2

stdpersecond = np.array((.4, .4, .1, .2, .15, 1.7, .3))
dt = .1
#stdmsmt = np.array((.6, .6, .3, .6, .3)) ### for voxeljones
stdmsmt = np.array((.4, .4, .2, .4, .2)) ### for prcnn
# P/(std^2 * dt) * (P - std^2*dt)
angvelredirect = 6.69#9 * stdpersecond[6]**2


nft = 56
piover2 = np.pi/2.

def uniformMeanAndVar(loval, hival):
    return (hival+loval)/2., (hival-loval)**2/12.

diagonal_idxs_sample = (np.arange(7), np.arange(7)) # handy for changing diagonals
diagonal_idxs_msmt = (np.arange(5), np.arange(5))

cov_msmt = np.diag(stdmsmt**2)
def prepMeasurement(msmt):
    return (np.array(msmt[:5]), cov_msmt)
    
def prepSample(sample):
    return sample[:5].copy(), sample[7:56].reshape((7,7))[:5,:5].copy()

"""
-log(integral_x p(x) p(z|x))
where p(x) and p(z|x) are normally distributed
this fn was coded considering msmts that may have different noise levels
if the noise is this same for each msmt (like in this file), likelihood could be
calculated a lot faster by precalculating the full variance for each object
still reaches acceptable speed via a gating check
"""
piterm = np.log(2*np.pi) * 5.
likelihood_zerovalue = 100.
likelihood_min_threshold = 4.**2
# radian offset in which a detection is considered possibly flipped
flip_angle_tol = .8
# extra log-cost for flipping orientation
flipped_log_cost = 2.
def likelihood(prepped_sample, msmt):
    xmean, xcov = prepped_sample[:2]
    zmean, zcov = msmt[:2]
    # early stopping
    deviation = zmean - xmean
    flipcost = 0.
    if deviation[2] > np.pi+flip_angle_tol:
        deviation[2] -= np.pi*2
    elif deviation[2] < -np.pi-flip_angle_tol:
        deviation[2] += np.pi*2
    elif deviation[2] > np.pi-flip_angle_tol:
        deviation[2] -= np.pi
        flipcost = flipped_log_cost
    elif deviation[2] < -np.pi+flip_angle_tol:
        deviation[2] += np.pi
        flipcost = flipped_log_cost
    variances = xcov[diagonal_idxs_msmt] + zcov[diagonal_idxs_msmt]
    if any(np.square(deviation) > variances*likelihood_min_threshold):
        return likelihood_zerovalue
    # kalman term for position variables
    # likelihood via decomp, assumes all eigenvalues nonzero
    eigvals, eigvecs = np.linalg.eigh(xcov + zcov)
    logdet = np.sum(np.log(eigvals))
    deviation_term = np.square(eigvecs.T.dot(deviation))
    linear_term = deviation_term.dot(1./eigvals)
    return (linear_term + logdet + piterm + flipcost)*.5
    

"""
"""
def update(sample, prepped_sample, msmt):
    sample_mean, sample_cov = prepped_sample[:2]
    msmt_mean, msmt_cov = msmt[:2]
    output = sample.copy()
    new_mean = output[:7]
    new_covs = output[7:56].reshape((7,7))
    # update positions and dimensions
    kalman_gain = np.linalg.solve(sample_cov + msmt_cov, new_covs[:5,:]).T
    dev = msmt_mean - sample_mean
    dev[2] = (dev[2]+np.pi)%(2*np.pi) - np.pi
    if dev[2] > np.pi - .8: dev[2] -= np.pi
    elif dev[2] < -np.pi + .8: dev[2] += np.pi
    new_mean += np.dot(kalman_gain, dev)
    new_covs -= np.dot(kalman_gain, new_covs[:5,:])
    # fix symmetry errors in length and width
    if new_mean[3] < 0:
        new_covs[3,:] *= -1
        new_covs[:,3] *= -1
        new_mean[3] *= -1
    if new_mean[4] < 0:
        new_covs[4,:] *= -1
        new_covs[:,4] *= -1
        new_mean[4] *= -1
    # standardize angle
    new_mean[2] = (new_mean[2] + np.pi) % (np.pi*2) - np.pi
    return output

"""
covariances of the next step's position were calculated using moment approximations
of the angle, for example cos(theta+del) = cos(theta)*(1-del^2/2) - sin(theta)*del

"""
varpertimestep = stdpersecond**2 * dt
def predict(sample):
    cos = np.cos(sample[2])
    sin = np.sin(sample[2])
    v = sample[5]
    cov = sample[7:56].reshape((7,7))
    covaa = cov[2,2]
    covav = cov[2,5]
    covvv = cov[5,5]
    # move mean
    sample[0] += cos * v * dt * (1-covaa/2) - sin*dt*covav
    sample[1] += sin * v * dt * (1-covaa/2) + cos*dt*covav
    sample[2] += sample[6] * dt
    # update covariance, 1st order
    cov[0,:] += cos*dt*cov[6,:] - sin*dt*v*cov[2,:]
    cov[1,:] += sin*dt*cov[6,:] + cos*dt*v*cov[2,:]
    cov[2,:] += cov[6,:]*dt
    cov[:,0] += cos*dt*cov[:,6] - sin*dt*v*cov[:,2]
    cov[:,1] += sin*dt*cov[:,6] + cos*dt*v*cov[:,2]
    cov[:,2] += cov[:,6]*dt
    # update covariance, 2nd order terms
    moment = (covav*covav + covaa*covvv) * dt * dt
    cov[0,0] += sin*sin*moment
    cov[1,1] += cos*cos*moment
    cov[0,1] -= cos*sin*moment
    cov[1,0] = cov[0,1]
    # add noise
    cov[diagonal_idxs_sample] += varpertimestep
    # flip speed + heading if motion is definitely fast backwards
    # don't need to flip heading covariance or angular velocity
    if v < -2 - 2*covvv**.5:
        sample[5] *= -1
        sample[2] = sample[2] % (2*np.pi) - np.pi
        cov[:,5] *= -1
        cov[5,:] *= -1
    # add pseudo-measurement to keep angular velocity near zero
    angvelprec = 1./(cov[6,6] + angvelredirect)
    sample[:7] -= angvelprec * sample[6] * cov[6]
    cov -= angvelprec * cov[6,:,None] * cov[6]
        
    
    
"""
the highest likelihood score that an object can get from this measurement
input: prepped msmt
output: likelihood (not negative log likelihood)
"""
nllnewobject = np.log(np.prod(stdmsmt)) # xyalw stdev
# as if this msmt were created by an object with one previous msmt
nllnewobject += 5./2*np.log(2)
nllnewobject += 5./2*np.log(2*np.pi)
#nllnewobject -= np.log(.1) # fp poisson process rate
def likelihoodNewObject(msmt): return nllnewobject


"""
the hypothetical sample that maximizes the likelihood of this msmt
used as a new measurement, from the 
motion is set to 0 with high variance
"""
initialspeedvariance = 5. ** 2
initialangvelvariance = .5 ** 2
def mlSample(msmt):
    mean, cov = msmt[:2]
    sample = np.zeros(56)
    sample[:5] = mean
    cov2 = sample[7:56].reshape((7,7))
    cov2[:5,:5] = cov
    cov2[5,5] = initialspeedvariance
    cov2[6,6] = initialangvelvariance
    return sample

def validSample(sample):
    valid = sample[0] > -30
    valid &= sample[0] < 100
    valid &= abs(sample[1]) < 100
    valid &= sample[3] > 0
    valid &= sample[3] < 20
    valid &= sample[4] > 0
    valid &= sample[4] < 20
    cov = sample[7:56].reshape((7,7))
    cov /= 2.
    cov += cov.T # symmetrize
    valid &= np.all(cov[diagonal_idxs_sample] > 0)
    valid &= np.linalg.det(cov) > 0
    valid &= cov[0,0] < 400#64
    valid &= cov[1,1] < 400#64
    valid &= cov[2,2] < 1.#.49
    valid &= cov[3,3] < 36#16
    valid &= cov[4,4] < 20#9
    return valid

"""
host vehicle moves, move and rotate sample
newpose = 2x3 rotation&translation matrix
"""
def reOrient(sample, newpose):
    sample[:2] = newpose[:2,:2].dot(sample[:2]) + newpose[:2,2]
    sample[2] += atan2(newpose[1,0], newpose[0,0])
    cov = sample[7:56].reshape((7,7)).copy()
    cov[:2,:] = newpose[:2,:2].dot(cov[:2,:])
    cov[:,:2] = cov[:,:2].dot(newpose[:2,:2].T)
    
"""
mean and covariance of object position
"""
def positionDistribution(sample):
    return sample[[0,1,7,15,8]]

def report(sample):
    return sample[:5].copy()
    


""" test on a single object in a single scene """    
if __name__ == '__main__':
    from imageio import imread
    from cv2 import imshow, waitKey, destroyWindow
    
    from plotStuff import plotImgKitti, addRect2KittiImg, plotRectangleEdges
    from calibs import calib_extrinsics, calib_projections, view_by_day
    from analyzeGT import readGroundTruthFileTracking
    from selfpos import loadSelfTransformations
    from presavedSensorPRCNN import getMsmts
    
    lidar_files = '/home/m2/Data/kitti/tracking_velodyne/training/{:04d}/{:06d}.bin'
    img_files = '/home/m2/Data/kitti/tracking_image/training/{:04d}/{:06d}.png'
    gt_files = '/home/m2/Data/kitti/tracking_gt/{:04d}.txt'
    oxt_files = '/home/m2/Data/kitti/oxts/{:04d}.txt'
    scene_idx = 2
    calib_idx = 0
    startfileidx = 87
    endfileidx = 130
    objid = 2

    def clear(): destroyWindow('a')
    
    calib_extrinsic = calib_extrinsics[calib_idx].copy()
    calib_projection = calib_projections[calib_idx]
    calib_intrinsic = calib_projection.dot(np.linalg.inv(calib_extrinsic))
    calib_extrinsic[2,3] += 1.65
    view_angle = view_by_day[calib_idx]
    with open(gt_files.format(scene_idx), 'r') as fd: gtfilestr = fd.read()
    gt_all, gtdc = readGroundTruthFileTracking(gtfilestr, ('Car', 'Van'))
    selfpos_transforms = loadSelfTransformations(oxt_files.format(scene_idx))
    
    sample = np.zeros(nft)
    previoussample = sample.copy()
    samplenotset = True
    
    for file_idx in range(startfileidx, endfileidx):
        img = imread(img_files.format(scene_idx, file_idx))[:,:,::-1]
        selfpos_transform = selfpos_transforms[file_idx][[0,1,3],:][:,[0,1,3]]
        gt = gt_all[file_idx]
        for gtobj in gt:
            if gtobj['id'] == objid: break
        haveobject = gtobj['id'] == objid
        
        # propagate sample
        if not samplenotset:
            previoussample[:] = sample
            reOrient(sample, selfpos_transform)
            assert validSample(sample)
            assert sample[0] > -5 and sample[0] < 70 and abs(sample[1]) < 50
            previoussample[:] = sample
            predict(sample)
            assert validSample(sample)
            assert sample[0] > -5 and sample[0] < 70 and abs(sample[1]) < 50
        
        # generate fake msmt
        msmts = getMsmts(scene_idx, file_idx)
        distances = np.hypot(msmts[:,0]-gtobj['box'][0],msmts[:,1]-gtobj['box'][1])
        havemsmt = haveobject & np.min(distances) < 7
        if havemsmt:
            msmt = msmts[np.argmin(distances),:5].copy()
            msmtprepped = prepMeasurement(msmt)
            if samplenotset:
                sample = mlSample(msmtprepped)
                samplenotset = False
            else:
                # determine msmt probability from sample vs from new object
                prepped_sample = prepSample(sample)
                llfromsample = likelihood(prepped_sample, msmtprepped)
                llfromnew = nllnewobject
                print("match vs miss loglik {:.2f}".format(llfromsample-llfromnew))
                assert llfromsample - llfromnew < 10
                # update sample
                sample = update(sample, prepped_sample, msmtprepped)
                
        validSample(sample)
        assert sample[0] > -5 and sample[0] < 70 and abs(sample[1]) < 50
        #print(sample[6])
        
        plotimg = plotImgKitti(view_angle)
        
        # draw object
        if haveobject:
            box = np.array(gtobj['box'])
            addRect2KittiImg(plotimg, box, (0,0,256,1.))
        # plot fake measurement
        if havemsmt:
            box = msmt.copy()
            plotRectangleEdges(plotimg, box, (0,256*.8,0,.8))
        # plot tracked sample
        if not samplenotset:
            box = sample[:5].copy()
            addRect2KittiImg(plotimg, box, (256*.4,0,0,.4))

        plotimg = np.minimum((plotimg[:,:,:3]/plotimg[:,:,3:]),255.).astype(np.uint8)
        # put the plot on top of the camera image to view, display for 3 seconds      
        display_img = np.zeros((plotimg.shape[0]+img.shape[0], img.shape[1], 3),
                               dtype=np.uint8)
        display_img[:plotimg.shape[0], (img.shape[1]-plotimg.shape[1])//2:
                    (img.shape[1]+plotimg.shape[1])//2] = plotimg
        display_img[plotimg.shape[0]:] = img
        imshow('a', display_img);
        if waitKey(300) == ord('q'):
            break