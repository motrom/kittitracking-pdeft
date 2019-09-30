# -*- coding: utf-8 -*-
"""
tracking details:
    objects have state, detectability, existence prob, and reality prob
    existence prob is for undetected / new objects
    reality prob affects detection probability, but only when view has moved
        reality determined in first step via score
"""

from runconfigs.example import lidar_files, img_files, gt_files, oxt_files, ground_files,\
                                save_estimates, estimate_files, display_video,\
                                save_video, video_file, scenes

import numpy as np
import numba as nb
from scipy.optimize import linear_sum_assignment
if display_video or save_video:
    from imageio import imread, get_writer
if display_video:
    from cv2 import imshow, waitKey

if display_video or save_video:
    from plotStuff import plotImgKitti, addRect2KittiImg, plotRectangleEdges, hsv2Rgba
    from kittiGT import readGroundTruthFileTracking
from calibs import calib_extrinsics, view_by_day
from grid import gridlen, gridstart, gridstep, floor
from gridtools import reOrientGrid, mixGrid, mapNormal2Subgrid
from selfpos import loadSelfTransformations
from subselectDetector import subselectDetector

from presavedSensorPRCNN import getMsmts as detect
import singleIntegrator as singleIntegrator
from occlusion import pointCloud2OcclusionImg, occlusionImg2Grid, boxTransparent
from hyphoplabeler import HypHopLabeler as Labeler

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
n_ft = singleIntegrator.nft
so_ft_pexist = singleIntegrator.ft_pexist


""" parameters for the tracker """
n_msmts_max = 100 # even sparse lidar detector doesn't report this many msmts
n_objects = 120 # higher - store less likely objects
deviation_to_recycle = 8. # meters
# This simulates only running the detector on certain regions,
# like track-before-detect. Unless you want to do that, set this to 1.
subselect_detector_ratio = 1.
# a stricter reporting cutoff can be used during evaluation
# this cutoff is primarily used to limit the number of reported estimates
report_score_cutoff = .1 

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
occupancy_move_buffer = 1.2 # meters
occupancydummy = np.zeros((gridlen[0]+occupancy_mixer.shape[0]-1,
                           gridlen[1]+occupancy_mixer.shape[1]-1))

## looks for empty matrices
emptydummy = np.zeros(gridlen+2, dtype=np.uint16)
emptymixer = np.array(((1,1,1),(1,1,1),(0,0,0)), dtype=np.uint16)
nabove3 = np.zeros(gridlen, dtype=np.uint16)
@nb.njit(nb.void(nb.f8[:,:], nb.f8[:,:,:], nb.u2[:,:]))
def _emptyTilesNb(data, ground, nabove):
    nabove[:] = 0
    for ptidx in range(data.shape[0]):
        pt = data[ptidx]
        tilex = int((pt[0]-gridstart[0])/gridstep[0])
        tiley = int((pt[1]-gridstart[1])/gridstep[1])
        if (tilex>=0 and tiley>=0 and tilex<gridlen[0] and tiley<gridlen[1]):
            height = np.dot(pt, ground[tilex, tiley,:3]) - ground[tilex,tiley,3]
            if height > .25:
                nabove[tilex,tiley] += 1
def findEmptyTiles(data, ground):
    _emptyTilesNb(data,ground,nabove3)
    mixGrid(nabove3, emptymixer, 10, emptydummy)
    return nabove3 < 2


# initialize state objects
objects = np.zeros((n_objects, n_ft))
newobjects = objects.copy()
matches = np.zeros((n_objects, n_msmts_max))
updatepairs = np.zeros((n_objects + n_msmts_max, 2), dtype=int)
prepruneweights = np.zeros(n_objects + n_msmts_max)

objectvisibilities = np.zeros(n_objects)
occupancy = np.zeros(gridlen) + occupancy_outer
occupancy_transform = np.eye(3)
visibility = occupancy.copy()
occlusionimg = None
labeler = Labeler(n_objects, 200, 10)

if save_video: video = get_writer(video_file, mode='I', fps=4)

for scene_idx, startfileidx, endfileidx, calib_idx in scenes:
    calib_extrinsic = calib_extrinsics[calib_idx]
    view_angle = view_by_day[calib_idx]
    with open(gt_files.format(scene_idx), 'r') as fd: gtfilestr = fd.read()
    gt_all, gtdontcares = readGroundTruthFileTracking(gtfilestr, ('Car', 'Van'))
    selfpos_transforms = loadSelfTransformations(oxt_files.format(scene_idx))
    
    objects[:] = 0.
    labeler.reset()
    
    for fileidx in range(startfileidx, endfileidx):
        # get data for this timestep
        data = np.fromfile(lidar_files.format(scene_idx, fileidx),
                           dtype=np.float32).reshape((-1,4))[:,:3]
        data = data.dot(calib_extrinsic[:3,:3].T) + calib_extrinsic[:3,3]
        img = imread(img_files.format(scene_idx, fileidx))[:,:,::-1]
        gt = gt_all[fileidx] # used for plotting only
        selfposT = selfpos_transforms[fileidx][[0,1,3],:][:,[0,1,3]]
        ground = np.load(ground_files.format(scene_idx, fileidx))
        
        # propagate objects
        for objidx in range(n_objects):
            if objects[so_ft_pexist]==0: continue
            obj = objects[objidx]
            soReOrient(obj, selfposT)
            soPredict(obj)
            # Recycle 'broken' objects,
            # for instance objects whose variance is too high to reliably associate
            # or calculate occlusion, etc accurately
            # recycle = treat as poisson and add to occupancy
            # This could be dangerous if it deletes actual vehicles!
            # In practice for the tried detectors, real vehicles are very rarely affected
            positiondist = soPositionDistribution(obj)
            xvv,yvv,xyv = positiondist[2:]
            maxstd = np.sqrt((xvv+yvv+np.hypot(xvv-yvv,2*xyv))/2)
            if maxstd > deviation_to_recycle:
                subgridloc, objsubgrid = mapNormal2Subgrid(positiondist,
                                            gridstart,gridstep,gridlen, subsize=2)
                subgridend = subgridloc + objsubgrid.shape
                occupancy[subgridloc[0]:subgridend[0],
                          subgridloc[1]:subgridend[1]] += objsubgrid * obj[so_ft_pexist]
                obj[so_ft_pexist] = 0.
        
        # propagate untracked object occupancy grid
        # to save time (and accuracy), only transform grid when host car has moved enough
        occupancy_transform = selfposT.dot(occupancy_transform)
        if abs(occupancy_transform[0,2]) > occupancy_move_buffer:
            occupancy = reOrientGrid(occupancy, occupancy_transform, occupancy_outer,
                                     gridstep, gridstart, gridlen)
            occupancy_transform = np.eye(3)
        # mix nearby tiles
        mixGrid(occupancy, occupancy_mixer, occupancy_outer, occupancydummy)
        occupancy += occupancy_constant_add
        occupancy[0,15:17] = 0. # this is the host car
        # determine occlusion and emptiness
        occlusionimg = pointCloud2OcclusionImg(data, occlusionimg)
        occlusionImg2Grid(occlusionimg, visibility, ground)
        
        # simulate only performing detection in certain regions
        # would be difficult to actually implement this for many detectors
        # but easy for VoxelJones detector
        empty = findEmptyTiles(data, ground)
        objecthypoprobabilities = np.ones(n_objects)
        tiles2detectgrid = subselectDetector(objects, objecthypoprobabilities,
                                             occupancy, visibility, empty,
                                             subselect_detector_ratio)
        visibility *= tiles2detectgrid
    
        # probability that a detection from this object is seen
        # (subselected & not occluded)
        for objidx in range(n_objects):
            if objects[objidx, so_ft_pexist] > 0:
                positiondist = soPositionDistribution(objects[objidx])
                subgridloc, occupysubgrid = mapNormal2Subgrid(positiondist,
                                                gridstart,gridstep,gridlen, subsize=2)
                subgridend = subgridloc + occupysubgrid.shape
                visibilitysubgrid = visibility[subgridloc[0]:subgridend[0],
                                               subgridloc[1]:subgridend[1]]
                objectvisibilities[objidx] = np.sum(occupysubgrid * visibilitysubgrid)
        objectvisibilities = np.minimum(objectvisibilities, 1)
        
        # get measurements
        msmts = detect(scene_idx, fileidx)
        # add context used by integrator -- aka occupancy
        # TODO remove boxTransparent
        msmtsnew = []
        for msmt in msmts:
            if boxTransparent(msmt, occlusionimg, ground):
                tilex, tiley = floor((msmt[:2]-gridstart)/gridstep).astype(int)
                if tiles2detectgrid[tilex,tiley]:
                    msmtsnew.append((msmt, occupancy[tilex, tiley]))
        msmts, msmtsold = msmtsnew, msmts

        # update occupancy grid, given detections
        occupancy *= 1-visibility
                    
        # prepare objs/msmts for data association
        nmsmts = len(msmts)
        msmtsprepped = [soPrepMsmt(msmt) for msmt in msmts]
        for objidx in range(n_objects):
            if not objects[objidx, so_ft_pexist]:
                matches[objidx,:nmsmts] = 100. # just some default high number
                continue
            objectprepped = soPrepObject(objects[objidx], objectvisibilities[objidx])
            for msmtidx in range(nmsmts):
                matches[objidx,msmtidx] = soLikelihood(objectprepped,
                                                       msmtsprepped[msmtidx])
        # data association
        matches[:,:nmsmts] = np.minimum(matches[:,:nmsmts], 0.)
        objmatch, msmtmatch = linear_sum_assignment(matches[:,:nmsmts])
        nmatches = len(objmatch)
        npairspreprune = n_objects + nmsmts - nmatches
        updatepairs[:n_objects,0] = np.arange(n_objects)
        updatepairs[:n_objects,1] = -1
        updatepairs[objmatch,1] = msmtmatch
        updatepairs[n_objects:npairspreprune,0] = -1
        msmtmisses = np.ones(nmsmts, dtype=bool)
        msmtmisses[msmtmatch] = False
        updatepairs[n_objects:npairspreprune,1] = np.where(msmtmisses)[0]    
        # prune
        nupdatepairs = min(n_objects, npairspreprune)
        for newobjidx in range(npairspreprune):
            objidx, msmtidx = updatepairs[newobjidx]
            if objidx == -1:
                prepruneweight = soPostMsmtMissWeight(msmts[msmtidx])
            elif msmtidx == -1:
                prepruneweight = soPostObjMissWeight(objects[objidx])
            else:
                prepruneweight = soPostMatchWeight(objects[objidx], msmts[msmtidx])
            prepruneweights[newobjidx] = prepruneweight
        prunedpairs = np.argpartition(prepruneweights[:npairspreprune],
                                     npairspreprune-n_objects)[npairspreprune-n_objects:]
        # tentative update
        for newobjidx in prunedpairs:
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
                   if newobjects[objidx, so_ft_pexist])
        objects, newobjects = newobjects, objects
        
        # report
        reportedobjects = []
        reportedscores = []
        reportedidxs = []
        for objidx in range(n_objects):
            reportscore, reportobj = soReport(objects[objidx])
            if reportscore > report_score_cutoff:
                reportedidxs.append(objidx)
                reportedobjects.append(reportobj)
                reportedscores.append(reportscore)
        reportedobjects = np.array(reportedobjects).reshape((-1,5))
        reportedscores = np.array(reportedscores)
        reportedlabels = labeler.add(updatepairs, reportedidxs)
        assert np.all(np.diff(np.sort(reportedlabels))) # all unique labels
        fullreports = np.column_stack((reportedobjects, reportedscores, reportedlabels))
#        fullreports = np.concatenate((reportedobjects, reportedscores[:,None],
#                                      reportedlabels[:,None]), axis=1)
        
        if save_estimates:
            np.save(estimate_files.format(scene_idx, fileidx), fullreports)
        
        # view
        if display_video or save_video:
            plotimg1 = plotImgKitti(view_angle)
            plotimg2 = plotImgKitti(view_angle)
            # shade tiles chosen for detection
            for tilex, tiley in np.ndindex(*gridlen):
                if not tiles2detectgrid[tilex,tiley] or empty[tilex,tiley]: continue
                tilecenterx = (tilex+.5)*gridstep[0] + gridstart[0]
                tilecentery = (tiley+.5)*gridstep[1] + gridstart[1]
                addRect2KittiImg(plotimg1, (tilecenterx, tilecentery, 0., 1.5, 1.5),
                                 np.array((30., 30., 30., .2)))
            # add ground truth
            for gtobj in gt:
                box = np.array(gtobj['box'])
                if gtobj['scored']:
                    addRect2KittiImg(plotimg1, box, (0,0,210*.9,.9))
                    addRect2KittiImg(plotimg2, box, (0,0,210*.9,.9))
                else:
                    addRect2KittiImg(plotimg1, box, (30*.9,80*.9,255*.9,.9))
                    addRect2KittiImg(plotimg2, box, (0,0,210*.9,.9))
            # add measurements
            for msmt in msmts:
                color = hsv2Rgba(.33, 1., .7, min(1., msmt[5]/.6))
                plotRectangleEdges(plotimg1, msmt[0], (70,255.9*.5,70,.5))
            # shade each tile by occupancy
            for tilex, tiley in np.ndindex(*gridlen):
                tilecenterx = (tilex+.5)*gridstep[0] + gridstart[0]
                tilecentery = (tiley+.5)*gridstep[1] + gridstart[1]
                if tilecenterx < 1 or tilecenterx > 58 or abs(tilecentery) > 28:
                    continue
                color = 255.9*(1-occupancy[tilex, tiley])
                #color = 255.9*visibility[tilex,tiley]#empty[tilex,tiley]#
                addRect2KittiImg(plotimg2, (tilecenterx, tilecentery, 0., 1.5, 1.5),
                                 np.array((color*.2, color*.2, color*.2, .2)))
            for report in fullreports:
                hue = report[6]%16/16.
                hue = (hue*5/6 + 9./12) % 1.
                shade = min(1., report[5] / .6)
                color = hsv2Rgba(hue, .9, 1., shade)
                plotRectangleEdges(plotimg2, report[:5], color)
            # put the plot on top of the camera image to view, display
            plotimg1 = np.minimum((plotimg1[:,:,:3]/plotimg1[:,:,3:]),
                                  255).astype(np.uint8)
            plotimg2 = np.minimum((plotimg2[:,:,:3]/plotimg2[:,:,3:]),
                                  255).astype(np.uint8)
            img = img[:368]
            display_img = np.zeros((640+img.shape[0], 1280, 3), dtype=np.uint8)
            display_img[:640, :640] = plotimg1
            display_img[:640, 640:] = plotimg2
            display_img[640:, (1280-img.shape[1])//2:(1280+img.shape[1])//2] = img
            
        if display_video:
            imshow('a', display_img);
            if waitKey(300) == ord('q'):
                break
        if save_video:
            video.append_data(display_img[:,:,::-1])
if save_video: video.close()