# -*- coding: utf-8 -*-
"""
last mod 2/4/19
"""

import numpy as np

lidar_files = '../object/training/velodyne/{:06d}.bin'
img_files = '../object/training/image_2/{:06d}.png'
gt_files = '../object/training_labels_orig/{:06d}.txt'
files_to_use = range(1000)

truncated_cutoffs = np.array((.15, .3, .5))
occluded_cutoffs = np.array((0, 1, 2))
height_cutoffs = np.array((40, 25, 25))
scored_classes = ('Car', 'Pedestrian', 'Cyclist')

nfiles_training = [154, 447, 233, 144, 314, 297, 270, 800, 390, 803, 294,
                   373, 78, 340, 106, 376, 209, 145, 339, 1059, 837]
        
def readGroundTruthFileTracking(gtstr, classes_include = ('Car',)):
    gtstr = gtstr.split('\n')
    if gtstr[-1] == '': gtstr.pop()
    nfiles = max(int(gtrow.split(' ')[0]) for gtrow in gtstr)+1
    outputs = [[] for file_idx in range(nfiles)]
    dontcares = [[] for file_idx in range(nfiles)]
    base_output = {'class':None,'box':None,'imbb':None,'difficulty':None,
                   'scored':None,'elevation':None,'id':None}
    for gtrow in gtstr:
        gtrow = gtrow.split(' ')
        file_idx = int(gtrow[0])
        # track id = int(gtrow[1])
        row_output = base_output.copy()
        row_output['id'] = int(gtrow[1])
        gtrow = gtrow[2:]
        this_class = gtrow[0]
        if this_class == "DontCare":
            dontcares[file_idx].append((float(gtrow[5]), float(gtrow[7]),
                                        float(gtrow[4]), float(gtrow[6])))
        if this_class not in classes_include:
            continue
        row_output['class'] = classes_include.index(this_class)
        # 2D rectangle on ground, in xyalw form
        gtang = 4.7124 - float(gtrow[14])
        gtang = gtang - 6.2832 if gtang > 3.1416 else gtang
        gtbox = (float(gtrow[13]), -float(gtrow[11]), gtang,
                 float(gtrow[10])/2, float(gtrow[9])/2)
        row_output['box'] = gtbox
        # 2D bounding box in image, top-bottom-left-right
        row_output['imbb'] = (float(gtrow[5]),float(gtrow[7]),
                              float(gtrow[4]),float(gtrow[6]))
        # elevation as meters up from car bottom
        row_output['elevation'] = 1.65-float(gtrow[12])
        # copy kitti's scoring-or-ignoring strategy
        truncation = float(gtrow[1])
        occlusion = int(gtrow[2])
        height = float(gtrow[7]) - float(gtrow[5]) # image bb height
        difficulty = 0
        for dd in range(3):
            not_met = truncation > truncated_cutoffs[dd]
            not_met |= occlusion > occluded_cutoffs[dd]
            not_met |= height < height_cutoffs[dd]
            if not_met: difficulty = dd + 1
        row_output['difficulty'] = difficulty
        row_output['scored'] = difficulty < 3 and this_class in scored_classes
        outputs[file_idx].append(row_output)
    return outputs, dontcares

def formatForKittiScoreTracking(ests, estids, scores, fileidx,
                                groundtiles, calib_project, imgshape, dontcares,
                                scorecutoff=.5, carheight=1.7):
    output_text_format = '{:d} {:d} {:s} {:.2f} {:d}' + ' {:.2f}'*13
    outputstr = []
    nests = len(ests)
    for estidx in range(nests):
        msmt = ests[estidx]
        if msmt[0]*.9 + 3 < abs(msmt[1]):
            continue
        if scores[estidx] < scorecutoff:
            continue
        tilex = min(19, max(0, int(msmt[0]/3+1)))
        tiley = min(31, max(0, int(msmt[1]/3+16)))
        groundtile = groundtiles[tilex, tiley]
        elev = groundtile[3] - groundtile[0]*msmt[0] - groundtile[1]*msmt[1]
        cos,sin = np.cos(msmt[2]), np.sin(msmt[2])
        corners = np.zeros((8,3))
        corners[0,:2] = msmt[:2] + (cos*msmt[3]+sin*msmt[4], sin*msmt[3]-cos*msmt[4])
        corners[1,:2] = msmt[:2] + (cos*msmt[3]-sin*msmt[4], sin*msmt[3]+cos*msmt[4])
        corners[2,:2] = msmt[:2] - (cos*msmt[3]+sin*msmt[4], sin*msmt[3]-cos*msmt[4])
        corners[3,:2] = msmt[:2] - (cos*msmt[3]-sin*msmt[4], sin*msmt[3]+cos*msmt[4])
        corners[:4,2] = elev
        corners[4:,:2] = corners[:4,:2]
        corners[4:,2] = elev + carheight
        msmt_corners = corners.dot(calib_project[:3,:3].T) + calib_project[:3,3]
        msmt_corners = msmt_corners[:,:2] / msmt_corners[:,2:]
        topfull, leftfull = np.min(msmt_corners, axis=0)
        bottomfull, rightfull = np.max(msmt_corners, axis=0)
        top, bottom, left, right = (max(topfull, 0), min(bottomfull, imgshape[0]),
                                    max(leftfull, 0), min(rightfull, imgshape[1]))
        imbb_area = (bottom - top) * (right - left)
        full_imbb_area = (bottomfull - topfull) * (rightfull - leftfull)
        truncation_level = 1 - imbb_area / full_imbb_area
        if truncation_level > .4:
            continue
        if bottom-top < 22:
            continue
        removed = False
        for dontcare in dontcares[fileidx]:
            overlap  = max(0, min(bottom-dontcare[0], dontcare[1]-top))
            overlap *= max(0, min(right-dontcare[2], dontcare[3]-left))
            overlap /= (bottom-top)*(right-left)
            if overlap > .6:
                removed = True
        if removed:
            continue
        observation_angle = np.pi/2. - np.arctan2(msmt[1], msmt[0])
        rotation_angle = np.pi/2. - msmt[2]
        estid = int(estids[estidx])
        score = scores[estidx]
        output = (fileidx,estid,'Car',0.,0,observation_angle,left,top,right,bottom,
                  1.7,#msmt[6],#
                  msmt[4]*2,msmt[3]*2,-msmt[1],-elev+1.65,msmt[0],
                  rotation_angle, score)
        #outputstr.append(output_text_format.format(*output))
    #return '\n'.join(outputstr)
        outputstr.append((msmt[:5].copy(), estid, score))
    return outputstr