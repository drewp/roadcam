from __future__ import division
import cv2, numpy, time

def over(target, targetRegion, source, sourceRegion, alpha):
    pano[targetRegion] *= (1 - alpha)
    pano[targetRegion] += source[sourceRegion] * alpha

def softColumnAlpha(shape, maxAlpha):
    alpha = numpy.ones(shape, numpy.float) * maxAlpha
    for col in range(alpha.shape[1]):
        scl = max(0, (1 - 2 * abs(col / alpha.shape[1] - .5)))
        alpha[:,col,:] = scl ** .7
    return alpha
    
def addWedgeToPano(wedge, offset, pano, maxAlpha=1):
    ox = int(offset[0,2])
    oy = int(offset[1,2])

    targetShape = pano[oy:oy+wedge.shape[0],
                       ox:(ox+wedge.shape[1]),
                       :].shape

    if not (targetShape[1] > 0 and targetShape[0] > 0):
        print "out of range, skip this wedge"
        return
        
    if ox <= 0:
        x1 = 0
        x2 = x1 + targetShape[1]
    else:
        x2 = wedge.shape[1]
        x1 = x2 - targetShape[1]
    if oy <= 0:
        y1 = 0
        y2 = y1 + targetShape[0]
    else:
        y2 = wedge.shape[0]
        y1 = y2 - targetShape[0]

    targetShape = (y2 - y1, x2 - x1, 3)
    alpha = softColumnAlpha(targetShape, maxAlpha)
    over(pano, (slice(max(0, oy),max(0, oy) + y2 - y1),
                slice(max(0, ox), max(0, ox) + x2 - x1),
                slice(None)),
         wedge, (slice(y1, y2),
                 slice(x1, x2),
                 slice(None)),
         alpha)

def extractWedge(frame, x, y, w, h, scl):
    return frame[y * scl:(y + h) * scl,
                 x * scl:(x + w) * scl]

def vidFrames(vid, step=1):
    num = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if frame is None:
            break
        num += 1
        if num % step != 0:
            continue
        yield num, frame

cv2.namedWindow('preview')
vid = cv2.VideoCapture('mEzTrOnPRic.mp4')#'Pinhurstclip.mov')

pano = numpy.zeros((900, 10000, 3), numpy.float)
prevWedge = None
offset = numpy.array([[1,0,0],[0,1,0]], numpy.float)
offset[0,2] = pano.shape[1]
lastWedgeX = None
lastPreviewTime = 0
imageScale = 1

for num, frame in vidFrames(vid, step=1):
    if imageScale == 1:
        small = frame
    else:
        small = cv2.resize(frame, None, fx=imageScale, fy=imageScale)
    
    wedge = extractWedge(small, 992, 2, 268, 687, imageScale)

    if prevWedge is not None:
        xform = cv2.estimateRigidTransform(prevWedge, wedge, False)
        if xform is not None:
            # never go backwards; it's an estimation error
            offset[0,2] -= max(0, xform[0,2])           
            offset[1,2] -= xform[1,2]

            returnScale = max(0, 1 - abs(xform[1,2]) / 80)
            offset[1,2] *= returnScale
            
            print "frame %s: offset now x=%s y=%s" % (
                num, offset[0,2], offset[1,2])
            if lastWedgeX is None or abs(offset[0,2] - lastWedgeX) > wedge.shape[1] * .2:
                addWedgeToPano(wedge, offset, pano)
                lastWedgeX = offset[0,2]
            if lastWedgeX < 5:
                print "filled the output"
                break
        
    prevWedge = wedge

    if time.time() > lastPreviewTime + .2 and lastWedgeX:
        print "drawing..."
        activeArea = pano[:,lastWedgeX-100:lastWedgeX + 500,:]
        active8Bit = numpy.clip(activeArea, 0, 255).astype(numpy.uint8)
        if min(active8Bit.shape) > 0:
            cv2.imshow('preview', active8Bit)
        ch = cv2.waitKey(10)
        lastPreviewTime = time.time()
        

print "saving..."
cv2.imwrite('out.png', pano)
