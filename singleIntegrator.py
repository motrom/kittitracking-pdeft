# -*- coding: utf-8 -*-
"""
handles things considered by the iPDA, or multi bernoulli filters
In this case,
    object existence - bernoulli probability
    object genuity - whether existing object is from long-term false positive
    object detectability - whether real object is temporarily undetectable
    object previous position x - for determining long-term false positives
    object previous position y - " "
    current number of detections
    average score of detections, used to determine, genuity
    label integer, (probably) unique for each originating msmt
    probability of being in detected tile (temporary for this timestep)
    object previous orientation cos
    object previous orientation sin
    stationaryness, used to determine genuity
"""
import numpy as np
from singleTracker import prepSample as soPrepObject
from singleTracker import prepMeasurement as soPrepMeasurement
from singleTracker import likelihood as soLikelihood
from singleTracker import update as soUpdate
from singleTracker import likelihoodNewObject as soLikelihoodNewObject
from singleTracker import mlSample as soNewObject
from singleTracker import validSample as soValidSample
from singleTracker import predict as soPredict
from singleTracker import reOrient as soReOrient
from singleTracker import positionDistribution as soPositionDistribution
from singleTracker import report as soReport
from singleTracker import nft as sonft

nft = sonft + 12
survival_probability = .997
detectability_steadystate = .99 # average probability of object being detected
detectability_timestep_ratio = .75 # amount it changes in one second
exist_object_ratio = .65
match_nll_offset = -2.


detectability_pos2pos = 1 - (1 - detectability_steadystate) * detectability_timestep_ratio
detectability_neg2pos = detectability_steadystate * detectability_timestep_ratio
detectability_multiplier = detectability_pos2pos - detectability_neg2pos
detectability_constant = detectability_neg2pos
def predict(sample):
    soPredict(sample[:sonft])
    pexist, preal, pdetect, origcenterx, origcentery = sample[sonft:sonft+5]
    newpexist = pexist*survival_probability
    # remove stuff outside of kitti visible zone
    if sample[0] < 0 or sample[0] > 55 or abs(sample[1]) > sample[0]*.87+1.87:
        newpexist *= .5
    if sample[0] < -5 or sample[0] > 63 or abs(sample[1]) > sample[0]*.87+6:
        newpexist = 0.
    # change reality based on absolute and relative motion
    currdist = np.hypot(sample[0],sample[1])
    prevdist = np.hypot(origcenterx, origcentery)
    assert min(currdist, prevdist) > 1e-8
    distanceratio = min(currdist, prevdist) / max(currdist, prevdist)
    distanceratio = min(distanceratio + .05, 1)
    sindist = abs(sample[sonft+10]*sample[0] - sample[sonft+9]*sample[1]) / currdist
    sindist = max(sindist - .05, 0)
    newstationaryness = distanceratio**2 * (1-sindist)**3
    newstationaryness *= 3.5 / max(abs(sample[5]), 3.5)
    newstationaryness = max(min(sample[sonft+11], newstationaryness), .3)
    sample[sonft+11] = newstationaryness
    newpreal = sample[sonft+6] /(sample[sonft+6]+
                                     (1-sample[sonft+6])*newstationaryness)
    existworks = preal / newpreal
    newpexist *= existworks / (1-newpexist+newpexist*existworks)
    newpdetect = pdetect*detectability_multiplier+detectability_constant
    sample[sonft:sonft+3] = (newpexist, newpreal, newpdetect)
    
def _nlOdds(p):
    if p < 1e-11: return 25.
    if 1-p < 1e-11: return -25.
    return -np.log(p / (1-p))
    
"""
returns:
    prepped object from single tracker
    negative log-odds of match (for adding to NLL of match matrix)
"""
def prepObject(sample, pindetectregion):
    sample[sonft+8] = pindetectregion # easy way to keep this around
    soprep = soPrepObject(sample[:sonft])
    pexist, preal, pdetect = sample[sonft:sonft+3]
    pmatches = pexist * pindetectregion
    return soprep + (_nlOdds(pmatches),)

"""
weights are for determining which objects to prune or report
only their relative values matter, but these are from 0 to 1
"""
def postMatchWeight(sample, msmt):
    preal = sample[sonft+1]
    realscore = msmt[2]
    preal = preal*realscore / (preal*realscore + (1-preal)*(1-realscore))
    return preal # pexist = 1

"""
prepObject must have been called at least once this timestep
    so that pindetectregion is updated
"""
def postObjMissWeight(sample):
    pexist, preal, pdetect, pindetectregion = sample[[sonft,sonft+1,sonft+2,sonft+8]]
    eer = pindetectregion
    existdetect = pdetect * eer
    pexist = pexist*(1-existdetect) / (pexist*(1-existdetect) + 1-pexist)
    return pexist * preal

"""
returns:
    prepped measurement from single tracker
    negative log-odds of msmt match (for adding to NLL of match matrix)
"""
def prepMeasurement(msmt):
    somsmt, newobjectrate, score = msmt
    soprep = soPrepMeasurement(somsmt)
    newmsmtlik = soLikelihoodNewObject(soprep)
    llexist = newobjectrate * exist_object_ratio
    llnotexist = (1-exist_object_ratio)
    logodds = np.log(llexist + llnotexist)
    return soprep + (logodds - newmsmtlik + match_nll_offset,)

def postMsmtMissWeight(msmt):
    somsmt, newobjectrate, score = msmt
    llexist = newobjectrate * exist_object_ratio
    llnotexist = (1-exist_object_ratio)
    pexist = llexist/ (llexist + llnotexist)
    return pexist*score

"""
actually negative log likelihood
this already accounts for false positive and negative probabilities
"""
def likelihood(preppedsample, preppedmsmt):
    solik = soLikelihood(preppedsample[:-1], preppedmsmt[:-1])
    return solik + preppedsample[-1] + preppedmsmt[-1]

def updateMatch(sample, msmt):
    preppedsample = soPrepObject(sample[:sonft])
    preppedmsmt = soPrepMeasurement(msmt[0])
    newsample = sample.copy()
    newsample[:sonft] = soUpdate(sample[:sonft], preppedsample, preppedmsmt)
    pexist, preal, pdetect, oldposx, oldposy = sample[sonft:sonft+5]
    newsample[sonft] = 1.
    # do correct update for reality score
    npreviousdetections = sample[sonft+5]
    meanscore = sample[sonft+6]
    realscore = msmt[2]
    if npreviousdetections < 10:
        newmeanscore = ((meanscore**2 *npreviousdetections + realscore**2)/(
                                                        npreviousdetections+1))**.5
    else:
        newmeanscore = (meanscore**2 * .9 + realscore**2 * .1)**.5
    newsample[sonft+1] = newmeanscore / (newmeanscore+
                                         (1-newmeanscore)*newsample[sonft+11])
    newsample[sonft+5] = npreviousdetections + 1
    newsample[sonft+6] = newmeanscore
    newsample[sonft+2] = 1.
    # update old positions for reality checks
    currdist = np.hypot(newsample[0], newsample[1])
    newsample[sonft+3] = (oldposx*4 + newsample[0])/5.
    newsample[sonft+4] = (oldposy*4 + newsample[1])/5.
    newsample[sonft+9] = (sample[sonft+9]*4 + newsample[0]/currdist)/5.
    newsample[sonft+10] = (sample[sonft+10]*4 + newsample[1]/currdist)/5.
    newsample[sonft+9:sonft+11] /= np.hypot(newsample[sonft+9],newsample[sonft+10])
    return newsample
    
def updateMiss(sample):
    newsample = sample.copy()
    pexist, preal, pdetect= sample[sonft:sonft+3]
    pindetectregion = sample[sonft+8]
    eer = pindetectregion
    existdetect = pdetect * eer
    newsample[sonft] = pexist*(1-existdetect) / (pexist*(1-existdetect) + 1-pexist)
    newsample[sonft+2] = pdetect*(1 - eer) / (pdetect*(1 - eer) + 1 - pdetect)
    return newsample

def updateNew(msmt):
    newsample = np.zeros(nft)
    newsample[:sonft] = soNewObject(soPrepMeasurement(msmt[0]))
    llexist = msmt[1] * exist_object_ratio
    llnotexist = 1 - exist_object_ratio
    newsample[sonft] = llexist/ (llexist + llnotexist)
    newsample[sonft+1] = msmt[2]
    newsample[sonft+2] = 1.
    newsample[sonft+3:sonft+5] = newsample[:2]
    newsample[sonft+5] = 1
    newsample[sonft+6] = newsample[sonft+1]
    newsample[sonft+7] = np.random.randint(int(1e9))#hash(newsample.data.tobytes()) % 1e9
    currdist = np.hypot(newsample[0], newsample[1])
    newsample[sonft+9] = newsample[0]/currdist
    newsample[sonft+10] = newsample[1]/currdist
    newsample[sonft+11] = 1.
    return newsample

def validSample(sample):
    valid = True
    if sample[sonft] > 1e-2:
        valid &= soValidSample(sample[:sonft])
    elif sample[sonft] > 1e-5 and not soValidSample(sample[:sonft]):
        print("invalid (but very unlikely) sample")
    valid &= sample[sonft] >= 0 and sample[sonft] <= 1
    valid &= sample[sonft+1] >= 0 and sample[sonft+1] <= 1
    valid &= sample[sonft+2] >= 0 and sample[sonft+2] <= 1
    valid &= sample[sonft+6] >= 0 and sample[sonft+6] <= 1
    return valid

def reOrient(sample, newpose):
    soReOrient(sample, newpose)
    sample[sonft+9:sonft+11] = np.dot(newpose[:2,:2], sample[sonft+9:sonft+11])
    
positionDistribution = soPositionDistribution


""" returns a float that is the prune weight, and the object in report format """
def report(sample):
    return sample[sonft]*sample[sonft+1], soReport(sample[:sonft])

""" if sample has 0 pexist, no reason to perform calculations (might be empty) """
def shouldUseObject(sample): return sample[sonft]*sample[sonft+1] > 0