# -*- coding: utf-8 -*-
"""
last mod 6/3/19
sample form:
    x center, y center, angle, len, wid, speed, flattened 6x6 cov of all
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

stdpersecond = np.array((.6, .6, .3, .2, .15, 1.6))
dt = .1
stdmsmt = np.array((.6, .6, .3, .6, .3))


nft = 42
piover2 = np.pi/2.

def uniformMeanAndVar(loval, hival):
    return (hival+loval)/2., (hival-loval)**2/12.

range_msmt = list(range(5)) # a handy list for indexing matrix diagonals

covmsmt = np.diag(stdmsmt**2)
def prepMeasurement(msmt):
    return (np.array(msmt), covmsmt)
    
def prepSample(sample):
    return sample[:5].copy(), sample[6:42].reshape((6,6))[:5,:5].copy()

"""
-2 * log(likelihood)
likelihood = integral_x p(x) p(z|x)
"""
piterm = np.log(2*np.pi) * 5.
likelihood_zerovalue = 100.
likelihood_min_threshold = 4.**2
flipped_log_cost = -9
def likelihood(prepped_sample, msmt):
    sample_mean, sample_cov = prepped_sample[:2]
    msmt_mean, msmt_cov = msmt[:2]
    # early stopping
    deviation = msmt_mean - sample_mean
    deviation[2] = (deviation[2]+np.pi) % (np.pi*2) - np.pi
    flipcost = 0.
    if deviation[2] > np.pi-.5:
        deviation[2] -= np.pi
        flipcost = flipped_log_cost
    elif deviation[2] < .5-np.pi:
        deviation[2] += np.pi
        flipcost = flipped_log_cost
    variances = sample_cov[range_msmt,range_msmt] + msmt_cov[range_msmt,range_msmt]
    if any(np.square(deviation) > variances*likelihood_min_threshold):
        return likelihood_zerovalue
    # kalman term for position variables
    # likelihood via decomp, assumes all eigenvalues nonzero (significant noise)
    eigvals, eigvecs = np.linalg.eigh(sample_cov + msmt_cov)
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
    new_mean = output[:6]
    new_covs = output[6:42].reshape((6,6))
    # update positions and dimensions
    kalman_gain = np.linalg.solve(sample_cov + msmt_cov, new_covs[:5,:]).T
    dev = msmt_mean - sample_mean
    dev[2] = (dev[2]+np.pi)%(2*np.pi) - np.pi
    if dev[2] > np.pi - .5: dev[2] -= np.pi
    elif dev[2] < -np.pi + .5: dev[2] += np.pi
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
    # move mean
    cos = np.cos(sample[2])
    sin = np.sin(sample[2])
    sample[0] += cos*sample[5] * dt
    sample[1] += sin*sample[5] * dt
    # update covariance
    cov = sample[6:42].reshape((6,6))
    covxa = cov[0,2]
    covxv = cov[0,5]
    covya = cov[1,2]
    covyv = cov[1,5]
    covaa = cov[2,2]
    covav = cov[2,5]
    covvv = cov[5,5]
    cos2 = cos*cos
    sin2 = sin*sin
    momenta4v2 = 3*covaa*covaa*covvv + 12*covaa*covav*covav
    momenta2v2 = covvv*covaa + 2*covav*covav
    covxvchange = dt*cos*(covvv - .5*momenta2v2)
    covxachange = dt*cos*(covav - 1.5*covav*covaa)
    covxxchange = 2*dt*cos*(covxv - .5*covxv*covaa - covxa*covav)
    covxxchange += dt*dt*(cos2*covvv + (sin2-cos2)*momenta2v2 + cos2*.25*momenta4v2)
    covyvchange = dt*sin*(covvv - .5*momenta2v2)
    covyachange = dt*sin*(covav - 1.5*covav*covaa)
    covyychange = 2*dt*sin*(covyv - .5*covyv*covaa - covya*covav)
    covyychange += dt*dt*(sin2*covvv + (cos2-sin2)*momenta2v2 + sin2*.25*momenta4v2)
    covxychange = dt*sin*(covxv - .5*covxv*covaa - covxa*covav)
    covxychange += dt*cos*(covyv - .5*covyv*covaa - covya*covav)
    covxychange += dt*dt*cos*sin*(covvv - 2*momenta2v2 + .25*momenta4v2)
    cov[0,0] += covxxchange
    cov[0,2] += covxachange
    cov[0,5] += covxvchange
    cov[1,1] += covyychange
    cov[1,2] += covyachange
    cov[1,5] += covyvchange
    cov[0,1] += covxychange
    cov[[1,2,5,2,5],[0,0,0,1,1]] = cov[[0,0,0,1,1],[1,2,5,2,5]] # symmetrize
    # add noise
    cov[range(6), range(6)] += varpertimestep
    # new 7/17/19 --- flip speed + heading if motion is definitively negative
    if (2.-sample[5]) < 2*sample[41]**.5:
        sample[5] *= -1
        sample[2] = sample[2] % (2*np.pi) - np.pi
        cov[:,5] *= -1
        cov[5,:] *= -1
        # don't think you need to flip angle covariance
        
    
    
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
the sample that maximizes the likelihood of this msmt - speed set to 0
"""
initialspeedstd = 5.
def mlSample(msmt):
    mean, cov = msmt[:2]
    sample = np.zeros(42)
    sample[:5] = mean
    cov2 = sample[6:42].reshape((6,6))
    cov2[:5,:5] = cov
    cov2[5,5] = initialspeedstd**2
    return sample

def validSample(sample):
    valid = sample[0] > -30
    valid &= sample[0] < 100
    valid &= abs(sample[1]) < 100
    valid &= sample[3] > 0
    valid &= sample[3] < 20
    valid &= sample[4] > 0
    valid &= sample[4] < 20
    cov = sample[6:42].reshape((6,6))
    cov /= 2.
    cov += cov.T # symmetrize
    valid &= np.all(cov[range(6), range(6)] > 0)
    valid &= np.linalg.det(cov) > 0
    valid &= cov[0,0] < 400#64
    valid &= cov[1,1] < 400#64
    valid &= cov[2,2] < 1.#.49
    valid &= cov[3,3] < 36#16
    valid &= cov[4,4] < 20#9
    return valid

"""
the POV is changed, move and rotate sample
"""
def reOrient(sample, newpose):
    F = np.eye(6)
    F[:2,:2] = newpose[:2,:2]
    sample[:2] = newpose[:2,:2].dot(sample[:2]) + newpose[:2,2]
    sample[2] += atan2(newpose[1,0], newpose[0,0])
    cov = sample[6:42].reshape((6,6)).copy()
    sample[6:42] = F.dot(cov).dot(F.T).reshape((36,))
    
def positionDistribution(sample):
    return sample[[0,1,6,13,7]]

def report(sample):
    return sample[:5].copy()
    


""" test on a single object in a single scene """    
if __name__ == '__main__':
    from imageio import imread
    from cv2 import imshow, waitKey, destroyWindow
    
    from plotStuff import base_image, plotRectangle, plotPoints, drawLine
    from plotStuff import plotImgKitti, addRect2KittiImg
    from calibs import calib_extrinsics, calib_projections, view_by_day
    from config import sceneranges
    from config import calib_map_training as calib_map
    from analyzeGT import readGroundTruthFileTracking
    from selfpos import loadSelfTransformations
    
    lidar_files = 'Data/tracking_velodyne/training/{:04d}/{:06d}.bin'
    img_files = 'Data/tracking_image/training/{:04d}/{:06d}.png'
    gt_files = 'Data/tracking_gt/{:04d}.txt'
    oxt_files = 'Data/oxts/{:04d}.txt'
    np.random.seed(0)
    scene_idx = 2
    objid = 2
    fake_noise = np.array((.6, .6, .3, .6, .25))
    fake_detect_prob = .8

    def clear(): destroyWindow('a')
    
    startfileidx, endfileidx = sceneranges[scene_idx]
    startfileidx = 87
    calib_idx = calib_map[scene_idx]
    calib_extrinsic = calib_extrinsics[calib_idx].copy()
    calib_projection = calib_projections[calib_idx]
    calib_intrinsic = calib_projection.dot(np.linalg.inv(calib_extrinsic))
    calib_extrinsic[2,3] += 1.65
    view_angle = view_by_day[calib_idx]
    with open(gt_files.format(scene_idx), 'r') as fd: gtfilestr = fd.read()
    gt_all = readGroundTruthFileTracking(gtfilestr, ('Car', 'Van'))
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
        havemsmtactually = gtobj['id'] == objid
        
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
        msmt = np.array(gtobj['box']) + np.random.normal(scale=fake_noise)
        havemsmt = havemsmtactually and np.random.rand() < fake_detect_prob
        if havemsmt:
            msmtprepped = prepMeasurement(msmt)
            if samplenotset:
                sample = mlSample(msmtprepped)
                samplenotset = False
            else:
            
                # determine msmt probability from sample vs from new object
                prepped_sample = prepSample(sample)
                llfromsample = likelihood(prepped_sample, msmtprepped)
                llfromnew = nllnewobject
                print(llfromsample - llfromnew)
                assert llfromsample - llfromnew < 10
            
                # update sample
                if llfromsample - llfromnew < 10:
                    sample = update(sample, prepped_sample, msmtprepped)
                
        validSample(sample)
        assert sample[0] > -5 and sample[0] < 70 and abs(sample[1]) < 50
        
        plotimg = plotImgKitti(view_angle)
        # draw object
        if havemsmtactually:
            box = np.array(gtobj['box'])
            addRect2KittiImg(plotimg, box, (0,0,256,1.))
        # plot fake measurement
        if havemsmt:
            box = msmt.copy()#[[1,2,0,3,4]].copy()
            addRect2KittiImg(plotimg, box, (0,256*.2,0,.2))
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
    #    imwrite(output_img_files.format(scene_idx, file_idx), display_img[:,:,::-1])
        imshow('a', display_img);
        if waitKey(300) == ord('q'):
            break