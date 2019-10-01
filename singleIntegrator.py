# -*- coding: utf-8 -*-
"""
handles things considered by the iPDA, or multi bernoulli filters
In this case,
    object existence - bernoulli probability
    object genuity - whether existing object is from long-term false positive
    object detectability - whether real object is temporarily undetectable
    current number of detections
    average score of detections, used to determine, genuity
    probability of being in detected tile (temporary for this timestep)
    stationaryness, used to determine genuity
    
takes objects from singleTracker
msmts like [x, y, a, l, w, detector score, untracked object rate]
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

# try alternate versions of the tracker
NOFAKES = False
NODETECT = False

nft = sonft + 7
ft_pexist = sonft
ft_genuity = sonft + 1
ft_detectability = sonft + 2
ft_detcount = sonft + 3
ft_detscore = sonft + 4
ft_stationaryness = sonft + 5
ft_visibility = sonft + 6


survival_probability = .997
detectability_steadystate = .99 # average probability of object being detected
detectability_timestep_ratio = .75 # amount it changes in one second
if NODETECT:
    detectability_timestep_ratio = .9999
exist_object_ratio = .65
match_nll_offset = -6.

detectability_pos2pos = 1 - (1 - detectability_steadystate) * detectability_timestep_ratio
detectability_neg2pos = detectability_steadystate * detectability_timestep_ratio
detectability_multiplier = detectability_pos2pos - detectability_neg2pos
detectability_constant = detectability_neg2pos
def predict(sample):
    soPredict(sample[:sonft])
    pexist, preal, pdetect = sample[sonft:sonft+3]
    newpexist = pexist*survival_probability
    # remove stuff outside of trackable zone
    if sample[0] < 0 or sample[0] > 55 or abs(sample[1]) > sample[0]*.87+1.87:
        newpexist *= .5
    if sample[0] < -5 or sample[0] > 63 or abs(sample[1]) > sample[0]*.87+6:
        newpexist = 0.
    # change reality based on motion
    newstationaryness = 3.5 / max(abs(sample[5]), 3.5)
    newstationaryness = max(min(sample[ft_stationaryness], newstationaryness), .3)
    sample[ft_stationaryness] = newstationaryness
    newpreal = sample[ft_detscore] /(sample[ft_detscore] +
                                     (1-sample[ft_detscore])*newstationaryness)
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
    sample[ft_visibility] = pindetectregion # easy way to keep this around
    soprep = soPrepObject(sample[:sonft])
    pexist, preal, pdetect = sample[sonft:sonft+3]
    pmatches = pexist*pdetect
    punmatch = 1 - pexist*pdetect*pindetectregion
    if pmatches < 1e-11: podds = 25.
    elif punmatch < 1e-11: podds = -25.
    else: podds = -np.log(pmatches / punmatch)
    return soprep + (podds,)

"""
weights are for determining which objects to prune or report
only their relative values matter, but these are from 0 to 1
"""
def postMatchWeight(sample, msmt):
    preal = sample[ft_genuity]
    realscore = msmt[5]
    preal = preal*realscore / (preal*realscore + (1-preal)*(1-realscore))
    return preal

"""
prepObject must have been called at least once this timestep
    so that pindetectregion is updated
"""
def postObjMissWeight(sample):
    pexist, preal, pdetect, pindetectregion = sample[[ft_pexist, ft_genuity,
                                                      ft_detectability,
                                                      ft_visibility]]
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
    soprep = soPrepMeasurement(msmt[:5])
    newmsmtlik = soLikelihoodNewObject(soprep)
    score, newobjectrate = msmt[5:]
    llexist = newobjectrate * exist_object_ratio
    if NOFAKES:
        llexist *= score
    llnotexist = (1-exist_object_ratio)
    logodds = np.log(llexist + llnotexist)
    return soprep + (logodds - newmsmtlik + match_nll_offset,)

def postMsmtMissWeight(msmt):
    score, newobjectrate = msmt[5:]
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
    preppedmsmt = soPrepMeasurement(msmt[:5])
    newsample = sample.copy()
    newsample[:sonft] = soUpdate(sample[:sonft], preppedsample, preppedmsmt)
    pexist, preal, pdetect, oldposx, oldposy = sample[sonft:sonft+5]
    newsample[ft_pexist] = 1.
    # do correct update for reality score
    npreviousdetections = sample[ft_detcount]
    meanscore = sample[ft_detscore]
    realscore = msmt[5]
    if NOFAKES:
        realscore = 1.
    if npreviousdetections < 10:
        newmeanscore = ((meanscore**2 *npreviousdetections + realscore**2)/(
                                                        npreviousdetections+1))**.5
    else:
        newmeanscore = (meanscore**2 * .9 + realscore**2 * .1)**.5
    newsample[ft_genuity] = newmeanscore / (newmeanscore+
                                         (1-newmeanscore)*newsample[ft_stationaryness])
    newsample[ft_detcount] = npreviousdetections + 1
    newsample[ft_detscore] = newmeanscore
    newsample[ft_detectability] = 1.
    return newsample
    
def updateMiss(sample):
    newsample = sample.copy()
    pexist, preal, pdetect= sample[sonft:sonft+3]
    pindetectregion = sample[ft_visibility]
    eer = pindetectregion
    existdetect = pdetect * eer
    newsample[ft_pexist] = pexist*(1-existdetect) / (pexist*(1-existdetect) + 1-pexist)
    newsample[ft_detectability] = pdetect*(1 - eer) / (pdetect*(1 - eer) + 1 - pdetect)
    return newsample

def updateNew(msmt):
    newsample = np.zeros(nft)
    newsample[:sonft] = soNewObject(soPrepMeasurement(msmt[:5]))
    score, newobjectrate = msmt[5:]
    llexist = newobjectrate * exist_object_ratio
    llnotexist = 1 - exist_object_ratio
    newsample[sonft:] = (llexist / (llexist + llnotexist), score, 1., 1, score,
                         1., 1.)
    if NOFAKES:
        newsample[sonft:] = (llexist/ (llexist + llnotexist) * msmt[2], 1., 1.,
                             1., 1., 1., 1.)
    return newsample

def validSample(sample):
    valid = True
    if sample[ft_pexist] > 1e-2:
        valid &= soValidSample(sample[:sonft])
    elif sample[ft_pexist] > 1e-5 and not soValidSample(sample[:sonft]):
        print("invalid (but very unlikely) sample")
    valid &= sample[ft_pexist] >= 0 and sample[ft_pexist] <= 1
    valid &= sample[ft_genuity] >= 0 and sample[ft_genuity] <= 1
    valid &= sample[ft_detectability] >= 0 and sample[ft_detectability] <= 1
    valid &= sample[ft_detscore] >= 0 and sample[ft_detscore] <= 1
    return valid

reOrient = soReOrient

positionDistribution = soPositionDistribution

""" returns a float that is the prune weight, and the object in report format """
def report(sample):
    return sample[ft_pexist]*sample[ft_genuity], soReport(sample[:sonft])