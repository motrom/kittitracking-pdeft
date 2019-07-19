# -*- coding: utf-8 -*-
"""
last modified 7/8/19
"""

import numpy as np
import numba as nb
np.random.seed(0) # randomness is only used to assign vehicle ids
from cv2 import imshow, waitKey, destroyWindow
from imageio import imread, get_writer

from presavedSensor import initializeSensor, getMsmts
from plotStuff import plotImgKitti, addRect2KittiImg, hsv2Rgba
from calibs import calib_extrinsics, view_by_day
from config import sceneranges
from config import calib_map_training as calib_map
from kittiGT import readGroundTruthFileTracking
from config import grndlen, grndstart, grndstep, floor
from occupancygrid import reOrientGrid, mixGrid, mapNormal2Subgrid
import singleIntegrator as singleIntegrator
soPrepObject = singleIntegrator.prepObject
soLikelihood = singleIntegrator.likelihood
soUpdateMatch = singleIntegrator.updateMatch
soUpdateMiss = singleIntegrator.updateMiss
soUpdateNew = singleIntegrator.updateNew
soPostMatchWeight = singleIntegrator.postMatchWeight
soPostObjMissWeight = singleIntegrator.postObjMissWeight
soPostMsmtMissWeight = singleIntegrator.postMsmtMissWeight
soPrepMsmt = singleIntegrator.prepMeasurement
validSample = singleIntegrator.validSample
soPredict = singleIntegrator.predict
soReOrient = singleIntegrator.reOrient
soPositionDistribution = singleIntegrator.positionDistribution
soReport = singleIntegrator.report
shouldUseObject = singleIntegrator.shouldUseObject
n_ft = singleIntegrator.nft
from selfpos import loadSelfTransformations
from mhtdaClink import mhtda, allocateWorkvarsforDA
from mhtdaClink import processOutput as mhtdaProcessOutput
from occlusion import pointCloud2OcclusionImg, occlusionImg2Grid, boxTransparent
from subselectDetector import subselectDetector
from evaluate import MetricAvgPrec as Metric #MetricPrecRec


lidar_files = 'Data/tracking_velodyne/training/{:04d}/{:06d}.bin'
img_files = 'Data/tracking_image/training/{:04d}/{:06d}.png'
gt_files = 'Data/tracking_gt/{:04d}.txt'
oxt_files = 'Data/oxts/{:04d}.txt'
#output_img_files = '../tracking/measurements/a/{:04d}/{:06d}.png'
ground_files = 'Data/tracking_ground/training/{:02d}f{:06d}.npy'
outestimatefiles = 'Data/estimates/trackingresults0/{:02d}f{:04d}.npy'
videofile = None#'resultsScene4.mp4'
scene_idx = 0 # 0 through 9

# tracker parameters
ntilesperstep = 260 # number of tiles to perform detection on
n_msmts_max = 100
n_objects = 120
n_hyps = 100


# visibility map
# update every timestep
# new obj prob map -- previous turns' visibility and new objs
# uncertainty map -- for each object in tile: exist prob, real prob, and variance
# propagate and update occupancy grid, given measurements and occlusion
occupancy_mixer = np.array([[.01, .04, .01],
                            [.04, .75, .04],
                            [.01, .04, .01]])
occupancy_constant_add = .05
occupancy_outer = 1. # starting new object rate
occupancydummy = np.zeros((grndlen[0]+occupancy_mixer.shape[0]-1,
                           grndlen[1]+occupancy_mixer.shape[1]-1))

def mhtdaPruneOutput(assocs, assocspruned, updatepairs, updatepairspruned,
                     prepruneweights, hypweights, nsols, nin, nout):
    hypweights2 = np.exp(hypweights[0] - hypweights[:nsols])
    prepruneweights[:nin] *= np.dot(hypweights2, assocs[:nsols,:nin])
    keepidxs = np.argpartition(prepruneweights[:nin], nin-nout)[nin-nout:]
    assocspruned[:] = False
    assocspruned[:nsols,:nout] = assocs[:nsols,keepidxs]
    updatepairspruned[:nout] = updatepairs[keepidxs]
    # just for debugging with score outside
    prepruneweights[:nout] = prepruneweights[keepidxs]

## looks for tiles with so few lidar points that they can't contain a vehicle
emptydummy = np.zeros(grndlen+2, dtype=np.uint16)
emptymixer = np.array(((1,1,1),(1,1,1),(0,0,0)), dtype=np.uint16)
nabove3 = np.zeros(grndlen, dtype=np.uint16)
@nb.njit(nb.void(nb.f8[:,:], nb.f8[:,:,:], nb.u2[:,:]))
def _emptyTilesNb(data, ground, nabove):
    nabove[:] = 0
    for ptidx in range(data.shape[0]):
        pt = data[ptidx]
        tilex = int(pt[0]/grndstep[0]-grndstart[0])
        tiley = int(pt[1]/grndstep[1]-grndstart[1])
        if (tilex>=0 and tiley>=0 and tilex<grndlen[0] and tiley<grndlen[1]):
            height = np.dot(pt, ground[tilex, tiley,:3]) - ground[tilex,tiley,3]
            if height > .25:
                nabove[tilex,tiley] += 1
def findEmptyTiles(data, ground):
    _emptyTilesNb(data,ground,nabove3)
    mixGrid(nabove3, emptymixer, 10, emptydummy)
    return nabove3 < 2




startfileidx, endfileidx = sceneranges[scene_idx]
#startfileidx = 115 # can set these to something else to crop scene
#endfileidx = 215
calib_idx = calib_map[scene_idx]
calib_extrinsic = calib_extrinsics[calib_idx].copy()
calib_extrinsic[2,3] += 1.65 # removing sensor's height
view_angle = view_by_day[calib_idx]
with open(gt_files.format(scene_idx), 'r') as fd: gtfilestr = fd.read()
gt_all, gtdontcares = readGroundTruthFileTracking(gtfilestr, ('Car', 'Van'))
selfpos_transforms = loadSelfTransformations(oxt_files.format(scene_idx))

# initialize state objects
objects = np.zeros((n_objects, n_ft))
associations = np.zeros((n_hyps, n_objects), dtype=np.bool8)
hypweights = np.zeros(n_hyps)
statebeforeupdate = (objects, associations, hypweights)
newobjects = objects.copy()
newassociations = associations.copy()
newhypweights = hypweights.copy()
stateafterupdate = (newobjects, newassociations, newhypweights)
# other initializations
n_objects_pre_prune = n_objects*4 + n_msmts_max*4
matches = np.zeros((n_objects, n_msmts_max))
msmtsubset = np.zeros((1,n_msmts_max), dtype=np.bool8)
msmtsubsetweights = np.zeros(1)
updatepairs = np.zeros((n_objects, 2), dtype=int)
updatepairspreprune = np.zeros((n_objects_pre_prune, 2), dtype=int)
associationspreprune = np.zeros((n_hyps, n_objects_pre_prune), dtype=np.bool8)
mhtdaworkvars = allocateWorkvarsforDA(n_objects, n_msmts_max, n_hyps)
mhtdaprocessindex = np.zeros((n_objects+1, n_msmts_max+1), dtype=int)
association_pairs = np.zeros((n_hyps, n_objects+n_msmts_max, 2), dtype=np.int32)
prepruneweights = np.zeros(n_objects_pre_prune)
nvalidhypotheses = 1 # initial hypothesis, no objects
objectdetectprobs = np.zeros(n_objects)
occupancy = np.zeros(grndlen) + occupancy_outer
occupancy_transform = np.eye(3)
visibility = occupancy.copy()
occlusionimg = None

sensorinfo = initializeSensor(scene_idx, startfileidx)
metric = Metric()
if videofile is not None: video = get_writer(videofile, mode='I', fps=4)

for fileidx in range(startfileidx, endfileidx):
    # get data for this timestep
    data = np.fromfile(lidar_files.format(scene_idx, fileidx),
                       dtype=np.float32).reshape((-1,4))[:,:3]
    data = data.dot(calib_extrinsic[:3,:3].T) + calib_extrinsic[:3,3]
    img = imread(img_files.format(scene_idx, fileidx))[:,:,::-1]
    gt = gt_all[fileidx]
    selfposT = selfpos_transforms[fileidx][[0,1,3],:][:,[0,1,3]]
    ground = np.load(ground_files.format(scene_idx, fileidx))
    
    # propagate objects
    for objidx in range(n_objects):
        obj = objects[objidx]
        if not shouldUseObject(obj): continue
        soReOrient(obj, selfposT)
        if obj[6] > 50 or obj[13] > 50:
            if obj[42] > .01:
                print("removed variant object at {:.0f},{:.0f} w/ exist {:.2f}".format(
                        obj[0], obj[1], obj[42]))
            # too uncertain of position to track well
            # just treat as poisson and add to occupancy
            tilex, tiley = np.floor(obj[:2]/grndstep).astype(int)-grndstart
            occupancy[max(tilex-2,0):tilex+3, max(tiley-2,0):tiley+3] += obj[42]/25.
            obj[42] = 0.
        soPredict(obj)
    hypweights -= np.min(hypweights) # prevent endless decrease in weights
    
    # propagate new object zones
    occupancy_transform = selfposT.dot(occupancy_transform)
    if abs(occupancy_transform[0,2]) > 1.2:
        occupancy = reOrientGrid(occupancy, occupancy_transform, occupancy_outer,
                                 grndstep, grndstart, grndlen)
        occupancy_transform = np.eye(3)
    # mix nearby tiles
    mixGrid(occupancy, occupancy_mixer, occupancy_outer, occupancydummy)
    occupancy += occupancy_constant_add
    occupancy[0,15:17] = 0. # this is your car
    # determine occlusion and emptiness
    occlusionimg = pointCloud2OcclusionImg(data, occlusionimg)
    occlusionImg2Grid(occlusionimg, visibility, ground)
    empty = findEmptyTiles(data, ground)
    
    # get measurements
    simmsmts = getMsmts(sensorinfo)
    # remove boxes that are considered unlikely to be correct, based on transparency
    includesimmsmts = np.zeros(simmsmts.shape[0],dtype=bool)
    for simmsmtidx in range(simmsmts.shape[0]):
        includesimmsmts[simmsmtidx] = boxTransparent(simmsmts[simmsmtidx],
                                                       occlusionimg, ground)
    simmsmts = simmsmts[includesimmsmts].copy()
    
    # simulate detection in certain regions rather than whole area
    # obviously, this is easier to implement for some detectors than for others
    hypothesis_probabilities = np.exp(-hypweights[:nvalidhypotheses])
    hypothesis_probabilities /= np.sum(hypothesis_probabilities)
    objecthypoprobabilities = np.dot(hypothesis_probabilities,
                                     associations[:nvalidhypotheses,:])
    tiles2detectgrid = subselectDetector(objects, objecthypoprobabilities,
                                    occupancy, visibility, empty, ntilesperstep)

    # determine probability of detecting each object based on location visibility
    for objidx in range(n_objects):
        if shouldUseObject(objects[objidx]):
            positiondist = soPositionDistribution(objects[objidx])
            subgridloc, occupysubgrid = mapNormal2Subgrid(positiondist,
                                            grndstart,grndstep,grndlen, subsize=2)
            subgridend = subgridloc + occupysubgrid.shape
            visibilitysubgrid = visibility[subgridloc[0]:subgridend[0],
                                           subgridloc[1]:subgridend[1]]
            tiles2detectsubgrid = tiles2detectgrid[subgridloc[0]:subgridend[0],
                                                   subgridloc[1]:subgridend[1]]
            objectdetectprobs[objidx] = np.einsum(occupysubgrid, [0,1],
                                                  visibilitysubgrid, [0,1],
                                                  tiles2detectsubgrid, [0,1], [])
        else:
            objectdetectprobs[objidx] = 0. # shouldn't matter...
    assert np.all(objectdetectprobs < 1+1e-8)
    objectdetectprobs = np.minimum(objectdetectprobs, 1)
    msmts = []
    for msmt in simmsmts:
        tilex, tiley = floor(msmt[:2]/grndstep).astype(int)-grndstart
        if tiles2detectgrid[tilex,tiley]:# and not empty[tilex,tiley]:
            msmts.append((msmt[:5], occupancy[tilex, tiley], msmt[5]))
    occupancy *= 1-visibility*tiles2detectgrid
                
    # prepare objs/msmts for data association
    nmsmts = len(msmts)
    msmtsprepped = []
    for msmtidx, msmtstuff in enumerate(msmts):
        msmtprepped = soPrepMsmt(msmtstuff)
        msmtsprepped.append(msmtprepped)
    # data association
    for objidx in range(n_objects):
        if not shouldUseObject(objects[objidx]):
            matches[objidx,:nmsmts] = 100
            continue
        objectprepped = soPrepObject(objects[objidx], objectdetectprobs[objidx])
        for msmtidx in range(nmsmts):
            matches[objidx,msmtidx] = soLikelihood(objectprepped,
                                                   msmtsprepped[msmtidx])
    msmtsubset[0,:nmsmts] = True
    msmtsubset[0,nmsmts:] = False
    mhtda(matches, associations, hypweights, nvalidhypotheses,
          msmtsubset, msmtsubsetweights, association_pairs, newhypweights,
          mhtdaworkvars)
    nvalidhypotheses = sum(newhypweights < 1000)
    nupdatepairspreprune = mhtdaProcessOutput(updatepairspreprune,
                                              associationspreprune,
                                              association_pairs[:nvalidhypotheses],
                                              mhtdaprocessindex, n_objects_pre_prune)
    assert mhtdaprocessindex[-1,-1] == -1
    nupdatepairs = min(n_objects, nupdatepairspreprune)
    for newobjidx in range(nupdatepairspreprune):
        objidx, msmtidx = updatepairspreprune[newobjidx]
        if objidx == -1:
            prepruneweight = soPostMsmtMissWeight(msmts[msmtidx])
        elif msmtidx == -1:
            prepruneweight = soPostObjMissWeight(objects[objidx])
        else:
            prepruneweight = soPostMatchWeight(objects[objidx], msmts[msmtidx])
        prepruneweights[newobjidx] = prepruneweight
    mhtdaPruneOutput(associationspreprune, newassociations, updatepairspreprune,
                     updatepairs, prepruneweights, newhypweights,
                     nvalidhypotheses, nupdatepairspreprune, nupdatepairs)
    # tentative update
    for newobjidx in range(nupdatepairs):
        objidx, msmtidx = updatepairs[newobjidx]
        if objidx == -1:
            # new object
            newobjects[newobjidx] = soUpdateNew(msmts[msmtidx])
        elif msmtidx == -1:
            # undetected object
            newobjects[newobjidx] = soUpdateMiss(objects[objidx])
        else:
            # object updated by measurement
            msmt = msmtsprepped[msmtidx]
            newobjects[newobjidx] = soUpdateMatch(objects[objidx], msmts[msmtidx])
    # resolve update
    assert all(validSample(newobjects[objidx]) for objidx in range(nupdatepairs)
                if shouldUseObject(newobjects[objidx]))
    newobjects[nupdatepairs:,42] = 0. # make sure unused pairs don't remain
    statebeforeupdate, stateafterupdate = (stateafterupdate, statebeforeupdate)
    objects, associations, hypweights = statebeforeupdate
    newobjects, newassociations, newhypweights = stateafterupdate
    
    # report
    reportedobjects = []
    reportedscores = []
    reportedlabels = []
    for objidx in range(n_objects):
        if associations[0,objidx]:
            reportscore, reportobj = soReport(objects[objidx])
            yesreport = reportscore > .05
            yesreport &= (reportobj[0]>0) & (reportobj[0]<50)
            yesreport &= reportobj[0]*.9 > abs(reportobj[1]) + .5
            if reportscore > .05:
                reportedobjects.append(reportobj)
                reportedscores.append(reportscore)
                reportedlabels.append(objects[objidx, 49])
    reportedobjects = np.array(reportedobjects).reshape((-1,5))
    reportedscores = np.array(reportedscores)
    reportedlabels = np.array(reportedlabels)
    metric.add(np.array([gtobj['box'] for gtobj in gt]),
               np.array([gtobj['scored'] for gtobj in gt]),
               np.array([gtobj['difficulty'] for gtobj in gt]),
               reportedobjects, reportedscores)
    np.save(outestimatefiles.format(scene_idx, fileidx),
            np.concatenate((reportedobjects, reportedscores[:,None],
                            reportedlabels[:,None]), axis=1))
    
    # visualize
    plotimg1 = plotImgKitti(view_angle)
    plotimg2 = plotImgKitti(view_angle)
    # shade tiles chosen for detection
    for tilex, tiley in np.ndindex(*grndlen):
        if not tiles2detectgrid[tilex,tiley] or empty[tilex,tiley]: continue
        tilecenterx = (tilex+grndstart[0] + .5)*grndstep[0]
        tilecentery = (tiley+grndstart[1] + .5)*grndstep[1]
        addRect2KittiImg(plotimg1, (tilecenterx, tilecentery, 0., 1.5, 1.5),
                         np.array((30., 30., 30., .2)))
    # add ground truth
    for gtobj in gt:
        box = np.array(gtobj['box'])
        if gtobj['scored']:
            addRect2KittiImg(plotimg1, box, (0,0,210*.9,.9))
        else:
            addRect2KittiImg(plotimg1, box, (30*.9,80*.9,255*.9,.9))
    # add measurements
    for msmt in msmts:
        addRect2KittiImg(plotimg1, msmt[0], (45,255.9*.5,45,.5))
    # shade each tile by occupancy
    for tilex, tiley in np.ndindex(*grndlen):
        tilecenterx = (tilex+grndstart[0] + .5)*grndstep[0]
        tilecentery = (tiley+grndstart[1] + .5)*grndstep[1]
        if tilecenterx < 1 or tilecenterx > 58 or abs(tilecentery) > 28:
            continue
        color = 255.9*(1-occupancy[tilex, tiley])#visibility[tilex,tiley]#empty[tilex,tiley]#
        addRect2KittiImg(plotimg2, (tilecenterx, tilecentery, 0., 1.5, 1.5),
                         np.array((color*.2, color*.2, color*.2, .2)))
    # add estimates
    gggg = associations[0] & (objects[:,42]*objects[:,43] > .5)
    for objidx in range(n_objects):
        if not associations[0,objidx]: continue
        reportscore, obj = soReport(objects[objidx])
        label = (objects[objidx,49] % 500) / 500.
        if reportscore > .5:
            addRect2KittiImg(plotimg2, obj, hsv2Rgba(label,.9,1.,.8))
        elif reportscore > .1:
            addRect2KittiImg(plotimg2, obj, hsv2Rgba(label,1.,1.,.15))
    # put the plot on top of the camera image to view, display
    plotimg1 = np.minimum((plotimg1[:,:,:3]/plotimg1[:,:,3:]),255.).astype(np.uint8)
    plotimg2 = np.minimum((plotimg2[:,:,:3]/plotimg2[:,:,3:]),255.).astype(np.uint8)
    img = img[:368]
    display_img = np.zeros((640+img.shape[0], 1280, 3), dtype=np.uint8)
    display_img[:640, :640] = plotimg1
    display_img[:640, 640:] = plotimg2
    display_img[640:, (1280-img.shape[1])//2:(1280+img.shape[1])//2] = img
    imshow('a', display_img);
    if waitKey(100) == ord('q'):
        break
    if videofile is not None: video.append_data(display_img[:,:,::-1])
if videofile is not None: video.close()

if fileidx == endfileidx-1: # not stopped partway
    print("Average Precision @ easy, mod, hard, unscored")
    print(metric.calc())
#    destroyWindow('a')