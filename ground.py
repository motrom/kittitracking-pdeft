# -*- coding: utf-8 -*-
"""
approximates ground using lidar points
ground is parameterized as grid of tiles, each of which is a flat plane
__main__ code saves ground in .npy files for use by detectors/trackers

This is not a remotely efficient approach to ground estimation, though it
is more accurate than a single plane in some cases. The obvious speed-up is to
not recalculate the same tiles each timestep -- given that the car only moves fast enough
to reach a small set of new tiles. Or you could make a map...
"""

import numpy as np
from grid import gridstep, gridstart, gridlen, floor

max_road_slope = .1 # tangent
# slope ~= .15 , current highest kitti slope seems to be .998
min_road_normal = 1./(max_road_slope**2 + 1)

standard_height = 1.65 # rough height of kitti forward camera

# ground calculation algorithm parameters
min_npoints = 100
min_ratio_below = .1
init_quantile = .3
sac_thresh = .15
sac_niter = 8
cutoff_divider_constant = np.sqrt(1 + .1**2 * 4)
cutoff_height = .5


def tilePoints(pts, gridstart, gridstep, gridlen):
    """
    points sorted by tile
    allows for fast access of points within certain set of tiles
    returns: pts array reordered (copy), integer matrix of pts-indices for each tile
    """
    pts_grnd = floor((pts[:,:2] - gridstart)/gridstep)
    include = np.all(pts_grnd >= 0, axis=1) & np.all(pts_grnd < gridlen, axis=1)
    grndidx = gridlen[1] * pts_grnd[include,0] + pts_grnd[include,1]
    grndorder = np.argsort(grndidx)
    tileidxs = np.searchsorted(grndidx[grndorder], range(gridlen[0]*gridlen[1]+1))
    #tileidxs = tileidxs.reshape(gridlen)
    return pts[include][grndorder], tileidxs


def getGround(pts):
    ntx, nty = gridlen
    # divide points by tiles
    ntiles = ntx * nty
    tile_range = range(ntiles)
    
    # select points within relevant region
    cutoffheight = cutoff_height * cutoff_divider_constant - standard_height
    include = pts[:,2] - (abs(pts[:,1]) + pts[:,0])*.1 < cutoffheight
    include &= np.all(pts >= gridstart, axis=1)
    include &= np.all(pts <= gridstart + gridlen*gridstep, axis=1)
    # optionally subsample points
    pts = pts[include][::1]
    
    quantized_xy = floor((pts[...,:2] - gridstart)/gridstep)
    # find supertile indices for first layer
    quantized_layer1 = quantized_xy // 2
    pt_tiles1 = quantized_layer1[:,0]*nty//2 + quantized_layer1[:,1]
    pts_reorder1 = np.argsort(pt_tiles1)
    tile_idxs1 = np.searchsorted(pt_tiles1[pts_reorder1], range(ntiles//4+1))
    # find supertile indices for second layer
    quantized_layer2 = (quantized_xy - 1) // 2
    pt_tiles2 = quantized_layer2[:,0]*(nty//2-1) + quantized_layer2[:,1]
    pts_reorder2 = np.argsort(pt_tiles2)
    tile_idxs2 = np.searchsorted(pt_tiles2[pts_reorder2], range((ntx//2-1)*(nty//2-1)+1))
    # find plane for each supertile
    # supertiles are indexing by the tile that is their bottom left corner
    # note that half the tiles are not bottom left to any supertile
    bottom_left_planes = np.zeros((ntiles, 4))
    bottom_left_success = np.zeros(ntiles, dtype=bool)
    bottom_left_scores = np.zeros(ntiles)
    for tile in tile_range:
        tilex = tile // nty
        tiley = tile % nty
        if tilex % 2 != tiley % 2: # is not the bottom left of any supertile
            continue
        if tilex == ntx-1: # top layer is not bottom left for anything
            continue
        if tiley == nty-1:
            continue
        
        if tilex % 2 > 0: # second layer
            layertile = (tilex-1)//2 * (nty//2-1) + (tiley-1)//2
            pts_tile = pts[pts_reorder2[tile_idxs2[layertile]:tile_idxs2[layertile+1]]]
        else: # first layer
            layertile = tilex//2 * nty // 2 + tiley//2
            pts_tile = pts[pts_reorder1[tile_idxs1[layertile]:tile_idxs1[layertile+1]]]
        
        if pts_tile.shape[0] < min_npoints:
            continue
        
        # initialize by trying several flat planes
        minheight = np.min(pts_tile[:,2])
        maxheight = np.max(pts_tile[:,2])
        n_bins = int((maxheight-minheight)/init_quantile)+1
        if n_bins == 1:
            init_height = min(maxheight, minheight + sac_thresh)
        else:
            bincounts,_ = np.histogram(pts_tile[:,2], bins=n_bins, density=False)
            highest_count = max(bincounts)
            # choose the lowest height that has at least half as many points as
            # the most popular height
            chosen_bin = np.where(bincounts > highest_count/2)[0][0]
            quantile = (maxheight-minheight)/n_bins
            init_height = (chosen_bin + .5) * quantile + minheight
        plane = np.array((0,0,1.,init_height))

        best_npoints = 0
        best_plane = np.array((0,0,1.,standard_height))
        for attempt in range(sac_niter):
            errors = abs(plane[3] - pts_tile.dot(plane[:3]))
            assert not np.any(np.isnan(errors))
            inliers = errors < sac_thresh
            npoints = sum(inliers)
            if npoints > best_npoints:
                best_plane = plane
                best_npoints = npoints
            elif npoints == 0:
                break # can't work off of no points
            
            assert pts_tile[inliers].shape[0] > 0
            meanvals = np.mean(pts_tile[inliers], axis=0)
            residuals = pts_tile[inliers] - meanvals
            covmatrix = residuals.T.dot(residuals)
            eigvals, eigvecs = np.linalg.eigh(covmatrix)
            normal = eigvecs[:, np.argmin(eigvals)]
            if normal[2] < 0:
                normal *= -1
            if normal[2] < min_road_normal: # too steep to be road
                break
            plane = np.append(normal, meanvals.dot(normal))
        
        plane = best_plane
        bottom_left_planes[tile] = plane
        points_below = sum(plane[3] - pts_tile.dot(plane[:3]) - sac_thresh > 0)
        ratio_below = float(points_below) / pts_tile.shape[0]
        include = ratio_below < min_ratio_below and best_npoints > min_npoints
        if include:
            bottom_left_success[tile] = True
            bottom_left_scores[tile] = best_npoints * (1 - ratio_below)
            
    # for each tile, determine which of two supertiles is the best fit
    planes = np.zeros((ntx, nty, 4))
    scores = np.zeros((ntx, nty)) + ntx + nty
    for tile in tile_range:
        tilex = tile // nty
        tiley = tile % nty
        if tilex % 2 == tiley % 2:
            # this tile is bottom left of one supertile and top right of another
            temp_score = 0
            if tilex < ntx-1 and tiley < nty-1: # is a bottom left
                if bottom_left_success[tile]:
                    planes[tilex, tiley] = bottom_left_planes[tile]
                    scores[tilex, tiley] = 0
                    temp_score = bottom_left_scores[tile]
            if tilex > 0 and tiley > 0: # is a top right
                adjtile = tile - nty - 1
                newscore = bottom_left_scores[adjtile]
                if bottom_left_success[adjtile] and newscore > temp_score:
                    planes[tilex, tiley] = bottom_left_planes[adjtile]
                    scores[tilex, tiley] = 0
        else:
            # this tile is bottom right and top left
            temp_score = 0
            if tilex > 0 and tiley < nty-1: # is a top left
                adjtile = tile - nty
                if bottom_left_success[adjtile]:
                    planes[tilex, tiley] = bottom_left_planes[adjtile]
                    scores[tilex, tiley] = 0
                    temp_score = bottom_left_scores[adjtile]
            if tilex < ntx-1 and tiley > 0: # is a bottom right
                adjtile = tile - 1
                newscore = bottom_left_scores[adjtile]
                if bottom_left_success[adjtile] and newscore > temp_score:
                    planes[tilex, tiley] = bottom_left_planes[adjtile]
                    scores[tilex, tiley] = 0
    
    # for yet-unfit tiles, determine which tiles to replace with
    # forward-right pass
    for tile in tile_range:
        # get nearby tiles and scores
        tilex = tile // nty
        tiley = tile % nty
        score = scores[tilex, tiley]
        if tilex == 0 and tiley == 0:
            adjacent_tiles = []
        elif tilex == 0:
            adjacent_tiles = [((tilex, tiley-1), 1)]
        elif tiley == 0:
            adjacent_tiles = [((tilex-1, tiley), 1)]
        else:
            adjacent_tiles = [((tilex, tiley-1), 1),
                              ((tilex-1, tiley), 1),
                              ((tilex-1, tiley-1), 1.5)]
        for adjacent_tile, penalty in adjacent_tiles:
            score2 = scores[adjacent_tile] + penalty
            if score2 < score:
                score = score2
                planes[tilex, tiley] = planes[adjacent_tile]
        scores[tilex, tiley] = score
    # backward-left pass    
    for tile in tile_range[::-1]:
        # get nearby tiles and scores
        tilex = tile // nty
        tiley = tile % nty
        score = scores[tilex, tiley]
        if tilex == ntx-1 and tiley == nty-1:
            adjacent_tiles = []
        elif tilex == ntx-1:
            adjacent_tiles = [((tilex, tiley+1), 1)]
        elif tiley == nty-1:
            adjacent_tiles = [((tilex+1, tiley), 1)]
        else:
            adjacent_tiles = [((tilex, tiley+1), 1),
                              ((tilex+1, tiley), 1),
                              ((tilex+1, tiley+1), 1.5)]
        for adjacent_tile, penalty in adjacent_tiles:
            score2 = scores[adjacent_tile] + penalty
            if score2 < score:
                score = score2
                planes[tilex, tiley] = planes[adjacent_tile]
        scores[tilex, tiley] = score
        
    return planes#, scores


def planes2Transforms(groundplanes):
    transforms = np.zeros((gridlen[0],gridlen[1],4,4))
    for tilex in range(gridlen[0]):
        for tiley in range(gridlen[1]):
            gridcenter = gridstart + gridstep*(tilex+.5,tiley+.5)
            plane = groundplanes[tilex, tiley]
            gridcenterz = plane[3] - plane[0]*gridcenter[0] - plane[1]*gridcenter[1]
            # find pose of tile, R.dot(pt in tile ref) + t = pt in global ref
            # know that R.dot([0,0,1]) = plane normal
            # and want to keep straight in BEV: R.dot([1,0,0]) = [c 0 s]
            planexz = np.hypot(plane[0], plane[2])
            planex = plane[0]/planexz
            planez = plane[2]/planexz
            T = np.array(((planez, -planex*plane[1], plane[0], gridcenter[0]),
                          (0., planexz, plane[1], gridcenter[1]),
                          (-planex, -planez*plane[1], plane[2], gridcenterz),
                          (0, 0, 0, 1)))
            # find transformation to tile reference
            # R.dot(pt) + t = pt in tile ref
            transforms[tilex, tiley] = np.linalg.inv(T)
    return transforms


"""
estimate ground for kitti tracking set, using scenes dictated in example
"""
if __name__ == '__main__':
    from os.path import isfile
    from calibs import calib_extrinsics
    
    from runconfigs.example import lidar_files, ground_files, scenes   
    
    for scene_idx, startfileidx, endfileidx, calib_idx in scenes:
        calib_extrinsic = calib_extrinsics[calib_idx].copy()
        print("making ground for scene {:d}".format(scene_idx))
        for fileidx in range(startfileidx, endfileidx):
            # load relevant data
            lidarfile = lidar_files.format(scene_idx, fileidx)
            if not isfile(lidarfile):
                print("couldn't find {:d}/{:d}, skipping".format(scene_idx,fileidx))
                continue
            data = np.fromfile(lidarfile, dtype=np.float32).reshape((-1,4))[:,:3]
            # get ground
            full_data_xyz = data.dot(calib_extrinsic[:3,:3].T)
            ground = getGround(full_data_xyz + calib_extrinsic[:3,3])
            np.save(ground_files.format(scene_idx, fileidx), ground)
            
         
            
""" visualize ground
"""
if False:
    from os.path import isfile
    from imageio import imread
    from cv2 import imshow, waitKey

    from grid import grnd2checkgrid
    from plotStuff import plotImgKitti
    from calibs import calib_extrinsics, calib_projections, view_by_day
    
    lidar_files = 'Data/tracking_velodyne/training/{:04d}/{:06d}.bin'
    ground_files = 'Data/tracking_ground/training/{:02d}f{:06d}.npy'
    img_files = 'Data/tracking_image/training/{:04d}/{:06d}.png'
    scene_idx = 1
    startfileidx = 181
    endfileidx = 447
    calib_idx = 0

    def grayer(img): return ((img.astype(float)-128)*.95 + 128).astype(np.uint8)
    
    calib_extrinsic = calib_extrinsics[calib_idx].copy()
    calib_projection = calib_projections[calib_idx]
    calib_intrinsic = calib_projection.dot(np.linalg.inv(calib_extrinsic))
    view_angle = view_by_day[calib_idx]
        
    for fileidx in range(startfileidx, endfileidx):
        lidarfile = lidar_files.format(scene_idx, fileidx)
        if not isfile(lidarfile):
            continue
        data = np.fromfile(lidarfile, dtype=np.float32).reshape((-1,4))[:,:3]
        data = data.dot(calib_extrinsic[:3,:3].T) + calib_extrinsic[:3,3]
        img = grayer(imread(img_files.format(scene_idx, fileidx))[:,:,::-1])
        ground = np.load(ground_files.format(scene_idx, fileidx))
        
        # shade by elevation
        plotimg = plotImgKitti(view_angle)
        max_elev = 5.
        pixel_to_ground = np.mgrid[40:640., :640]
        pixel_to_ground[0] *= -60. / 640
        pixel_to_ground[1] *= -60. / 640
        pixel_to_ground[0] += 60.
        pixel_to_ground[1] += 30.
        pixel_to_ground = pixel_to_ground.transpose((1,2,0))
        quantized = floor((pixel_to_ground[:,:,:2]-gridstart)/gridstep)
        planes = ground[quantized[:,:,0], quantized[:,:,1]]
        heights = (planes[:,:,3] - planes[:,:,0]*pixel_to_ground[:,:,0] -
                   planes[:,:,1]*pixel_to_ground[:,:,1])
        heights = np.maximum(np.minimum(heights, max_elev), -max_elev)
        plotimg[40:640,:,1] = 255 + np.minimum(heights,0)/max_elev*255
        plotimg[40:640,:,2] = 255 - np.maximum(heights,0)/max_elev*255
        plotimg = np.minimum((plotimg[:,:,:3]/plotimg[:,:,3:]),255.).astype(np.uint8)
        
        # add lidar points to image
        tpts, tidxs = tilePoints(data, gridstart, gridstep, gridlen)
        for tilex, tiley in grnd2checkgrid:
            tileidx = tilex*gridlen[1] + tiley
            pts = tpts[tidxs[tileidx]:tidxs[tileidx+1]]
            groundtile = ground[tilex, tiley]
            heights = pts.dot(groundtile[:3]) - groundtile[3]
            color = np.zeros((pts.shape[0], 3),dtype=np.uint8) + (0,0,255)
            color[heights < .3] = (0,255,0)
            color[heights < .1] = (255,0,0)
            pts = pts.dot(calib_intrinsic[:3,:3].T) + calib_intrinsic[:3,3]
            ptsinplot = (pts[:,:2] / pts[:,2:]).astype(int)
            include  = (ptsinplot[:,0]>=0) & (ptsinplot[:,0] < img.shape[0])
            include &= (ptsinplot[:,1]>=0) & (ptsinplot[:,1] < img.shape[1])
            img[ptsinplot[include,0],ptsinplot[include,1]] = color[include]
        
        display_img = np.zeros((plotimg.shape[0]+img.shape[0], img.shape[1], 3),
                               dtype=np.uint8)
        display_img[:plotimg.shape[0], (img.shape[1]-plotimg.shape[1])//2:
                    (img.shape[1]+plotimg.shape[1])//2] = plotimg
        display_img[plotimg.shape[0]:] = img
        imshow('a', display_img);
        if waitKey(1000) == ord('q'):
            break