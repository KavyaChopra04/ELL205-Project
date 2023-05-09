import sys
import cv2
import numpy
from scipy.signal import medfilt
from time import time
from cvxpy import *
from tqdm import tqdm
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

PLOT = True
COMPARE = True
MOTION_VECTORS = True

meshSize = 16     # size of block in mesh
def initialiseBlock(x, y, rowWidth, colWidth):
    mainBlock=[]
    for i in range(x, x+2):
        for j in range(y, y+2):
            mainBlock.append([i*colWidth, j*rowWidth])
    return mainBlock
def initialiseDistances(rowNumber, colNumber, x_motion_mesh, y_motion_mesh, rowWidth, colWidth):
    distances=[]
    for column in range(colNumber, colNumber+2):
        for row in range(rowNumber, rowNumber+2):
            distances.append([column*colWidth+x_motion_mesh[row, column], row*rowWidth+y_motion_mesh[row, column]])
    return distances
def insertEntry(x, dictval, val1, val2):
    if(x in dictval.keys()):
        dictval[x].append(val1-val2)
    else:
        dictval[x] = [val1-val2]
def addMedianFilter(mainDict, tempDict, meshDict):
    for key in list(mainDict.keys()):
        if(key in tempDict.keys()):
            tempDict[key].sort()
            meshDict[key] = mainDict[key]+tempDict[key][len(tempDict[key])//2]
        else:
            meshDict[key] = mainDict[key]
def dist(a, b):
    """
    Returns the Euclidean distance between two points.

    Parameters:
        a: The first point.
        b: The second point.
    
    Returns:
        The Euclidean distance between a and b.
    """

    return numpy.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
def transform(Homography_Matrix, Point_Vector):
    """
    Transforms a point using a homography matrix.

    Parameters:
        Homography_Matrix: The Homography Matrix, having dimension 3x3 
        Point_Vector: The Point Vector, having dimension 2x1 (x,y)
    
    Returns:
        A transformed point given by Homography_Matrix*Point_Vector.
    """

    a = Homography_Matrix[0,0]*Point_Vector[0] + Homography_Matrix[0,1]*Point_Vector[1] + Homography_Matrix[0,2]
    b = Homography_Matrix[1,0]*Point_Vector[0] + Homography_Matrix[1,1]*Point_Vector[1] + Homography_Matrix[1,2]
    c = Homography_Matrix[2,0]*Point_Vector[0] + Homography_Matrix[2,1]*Point_Vector[1] + Homography_Matrix[2,2]

    try:
        return [a/c, b/c]
    except:
        return Point_Vector
def inEllipse(vertexCoordinates, oldFeature, rowWidth, colWidth):

    return ((vertexCoordinates[0] - oldFeature[0])/rowWidth)**2 + ((vertexCoordinates[1] - oldFeature[1])/colWidth)**2 <= 90
def transfer_motion(old_FP, new_FP, frame):
    """
    Parameters:
        frame:  The frame to be warped.
        old_FP: The feature points in the previous frame.
        new_FP: The feature points in the current frame.
    
    Returns:
        A motion mesh in x and y direction for frame
    """
    meshSize = 16
    if(old_FP.shape[0] < 4):
        return [], []
    rowWidth = frame.shape[0] // meshSize
    colWidth = frame.shape[1] // meshSize
    base_x_motion = numpy.zeros((meshSize, meshSize), dtype=float)
    base_y_motion = numpy.zeros((meshSize, meshSize), dtype=float)
    Homography_Matrix, status = cv2.findHomography(old_FP, new_FP, cv2.RANSAC)
    combinedFrame = numpy.zeros((old_FP.shape[0], 4))
    propagation_x = {}
    propagation_y = {}
    for meshRow in range(meshSize):
        for meshCol in range(meshSize):
            propagation_x[(meshRow, meshCol)] =[]
            propagation_y[(meshRow, meshCol)] = []
    for i in range(old_FP.shape[0]):
        combinedFrame[i, 0] = old_FP[i,0]
        combinedFrame[i, 1] = old_FP[i,1]
        combinedFrame[i, 2] = new_FP[i,0]
        combinedFrame[i, 3] = new_FP[i,1]
    for mapping in combinedFrame:
        oldFeature = [mapping[0], mapping[1]]
        newFeature = [mapping[2], mapping[3]]
        transformOld = transform(Homography_Matrix, oldFeature)
        for meshRow in range(meshSize):
            for meshCol in range(meshSize):
                vertexCoordinates = [meshCol*colWidth, meshRow*rowWidth]
                if(inEllipse(vertexCoordinates, oldFeature, rowWidth, colWidth)):
                    propagation_x[(meshRow, meshCol)].append(mapping[2] - transformOld[0])
                    propagation_y[(meshRow, meshCol)].append(mapping[3] - transformOld[1])
    for meshRow in range(meshSize):
        for meshCol in range(meshSize):
            transformedVertex = transform(Homography_Matrix, [meshCol*colWidth, meshRow*rowWidth])
            base_x_motion[meshRow, meshCol] = meshCol*colWidth - transformedVertex[0]
            base_y_motion[meshRow, meshCol] = meshRow*rowWidth - transformedVertex[1]
    motion_mesh_x = numpy.zeros((meshSize, meshSize), dtype=float)
    motion_mesh_y = numpy.zeros((meshSize, meshSize), dtype=float)
    for meshRow in range(meshSize):
        for meshCol in range(meshSize):
            lst = sorted(propagation_x[(meshRow, meshCol)])
            if(propagation_x[(meshRow, meshCol)]!=[]):
                motion_mesh_x[meshRow, meshCol]= base_x_motion[meshRow, meshCol] + lst[len(propagation_x[(meshRow, meshCol)])//2]
            if(propagation_y[(meshRow, meshCol)]!=[]):
                motion_mesh_y[meshRow, meshCol]= base_y_motion[meshRow, meshCol] + lst[len(propagation_y[(meshRow, meshCol)])//2]
    motion_mesh_x = medfilt(motion_mesh_x, kernel_size=[3, 3])
    motion_mesh_y = medfilt(motion_mesh_y, kernel_size=[3, 3])
    return motion_mesh_x, motion_mesh_y
def update_profile(x_VP, y_VP, x_mesh, y_mesh):
    """
    Parameters:
        x_VP:        vertex profiles along x-direction
        y_VP:        vertex profiles along y-direction
        x_mesh:  motion mesh along x-direction
        y_mesh:  motion mesh along y-direction

    Returns:
        Appends x_mesh, y_mesh to x_VP, y_VP
    """

    tmp_y = y_VP[:, :, -1] + y_mesh     
    tmp_x = x_VP[:, :, -1] + x_mesh

    y_VP = numpy.concatenate((y_VP, numpy.expand_dims(tmp_y, axis=2)), axis=2)
    x_VP = numpy.concatenate((x_VP, numpy.expand_dims(tmp_x, axis=2)), axis=2)
    
    return x_VP, y_VP


def frame_warp(frame, x_motion_mesh, y_motion_mesh):
    """
    Parameters:
        frame:          The frame to be warped.
        x_motion_mesh:  The motion mesh in x direction.
        y_motion_mesh:  The motion mesh in y direction.

    Returns:
        A mesh warped frame according to given motion meshes x_motion_mesh, y_motion_mesh
    """
    frameRows = frame.shape[0]       # height of frame
    frameCols = frame.shape[1]        # width of frame
    rowWidth = frame.shape[0] // meshSize
    colWidth = frame.shape[1] // meshSize
    meshHeight = meshSize*rowWidth         # height of mesh
    meshWidth = meshSize*colWidth         # width of mesh
    horizontalArray = numpy.zeros((frameRows, frameCols), numpy.float32)    # map for x direction
    verticalArray = numpy.zeros((frameRows, frameCols), numpy.float32)    # map for y direction
    for meshRow in range(meshSize-1):
        for meshCol in range(meshSize-1):
            actualBlock = numpy.asarray(initialiseDistances(meshRow, meshCol, x_motion_mesh, y_motion_mesh, rowWidth, colWidth)) 
            meshBlock = numpy.asarray(initialiseBlock(meshCol,meshRow, rowWidth, colWidth))
            Homography_Matrix, _ = cv2.findHomography(meshBlock, actualBlock, cv2.RANSAC)
            for y_coord in range(rowWidth*meshRow, rowWidth*(meshRow+1)):
                for x_coord in range(colWidth*meshCol, colWidth*(meshCol+1)):
                    horizontalArray[y_coord, x_coord], verticalArray[y_coord, x_coord] = transform(Homography_Matrix, [x_coord, y_coord])    
    # fills the remaining part of horizontalArray and verticalArray
    for externalRow in range(meshHeight, horizontalArray.shape[0]):
            horizontalArray[externalRow, :] = horizontalArray[meshHeight-1, :]
            verticalArray[externalRow, :] = verticalArray[meshHeight-1, :]
    for externalCol in range(meshWidth, horizontalArray.shape[1]):
            horizontalArray[:, externalCol] = horizontalArray[:, meshWidth-1]
            verticalArray[:, externalCol] = verticalArray[:, meshWidth-1]
    # warps the frame
    mesh = cv2.remap(src=frame, map1=horizontalArray, map2=verticalArray, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return mesh



meshSize = 16     # size of block in mesh



def calculate_gaussian_weight(curr_index, point_index, window_size, temporal_smoothness_weight):
    """
    Calculates the spatial Gaussian weight for a given window size and point
    within the window.

    Parameters:
        curr_index:  the current index in the window
        point_index: the index of the point for which the Gaussian weight is being calculated
        window_size: the size of the window over which the Gaussian weight is applied

    Returns:
        The spatial Gaussian weight for the given window size and point.
    """

    if ((point_index - curr_index >= window_size + 1) or (curr_index - point_index >= window_size + 1)):
        return 0
    else:
        return numpy.exp(-1* ((point_index - curr_index)**2) / ((temporal_smoothness_weight * window_size)/3 )**2)



def optimize_trajectory(c):
    """
    solutionArrayarameters:
        c:           camera trajectory, a 3D numpy array of size (height, width, frames),
                     where height and width represent the dimensions of the camera frames and frames represents the number of camera frames. 
        buffer_size: size of buffer to be used for optimization
        iterations:  number of iterations to be used for optimization
        window_size: the size of window over which a Gaussian weighting
                     function is applied to impose spatial smoothness
        beta:        weight of temporal smoothness

    Returns:
        Optimized camera trajectory
    """
    print(c.shape)
    buffer_size=40 
    iterations=20 
    window_size=20
    temporal_smoothness_weight=1
    spatial_smoothness_weight=100  # weight of spatial smoothness
    optimizedTrajectory = numpy.empty_like(c)    # optimized camera trajectory initialization
    trajectoryRows=c.shape[0]
    trajectoryCols=c.shape[1]
    trajectoryFrames=c.shape[2]
    gaussianWeightList=numpy.zeros((buffer_size, buffer_size))    # spatial gaussian weights initialization
    
    for i in range(gaussianWeightList.shape[0]):
        for j in range(gaussianWeightList.shape[1]):
            gaussianWeightList[i,j] = calculate_gaussian_weight(i, j, window_size, temporal_smoothness_weight)
    for row in range(trajectoryRows):
        for col in range(trajectoryCols):
            """
                d:  used to store the previous set of optimized camera trajectory points. 
            """
            differenceTrajectory = None
            y = []
            # real-time optimization
            for frameNumber in range(1, trajectoryFrames+1):
                # If we haven't filled up the buffer yet, use all previous frames
                if frameNumber < buffer_size+1:
                    solutionArray = numpy.asarray(c[row, col, :frameNumber]) #current mesh vertex of current frame
                    if differenceTrajectory is not None:
                        for i in range(iterations):
                            """
                                currentSolution is a vector that represents the solution at each iteration of the solver. 
                                It is calculated by adding the product of the spatial Gaussian weights
                                and the camera trajectory values to the weighted sum of the 
                                previous solution vector and the temporal difference d (scaled by the temporal weight beta).
                            """
                            currentSolution = c[row, col, :frameNumber] + spatial_smoothness_weight*numpy.dot(gaussianWeightList[:frameNumber, :frameNumber], solutionArray)
                            currentSolution[:-1] = currentSolution[:-1] + temporal_smoothness_weight*differenceTrajectory #extract the last frame, and use it as the current solution to update the next one
                            """
                                diagonalElement is also a vector that represents the diagonal elements of the matrix in the Jacobi solver. 
                                It is calculated by adding the spatial Gaussian weights to the temporal weight beta.
                            """
                            diagonalElements = 1 + spatial_smoothness_weight*numpy.dot(gaussianWeightList[:frameNumber, :frameNumber], numpy.ones((frameNumber,)))
                            diagonalElements[:-1] = diagonalElements[:-1] + temporal_smoothness_weight #last diagonal element
                            solutionArray = numpy.divide(currentSolution, diagonalElements)
                else:
                    solutionArray = numpy.asarray(c[row, col, frameNumber-buffer_size:frameNumber]) 
                    for i in range(iterations):
                        currentSolution = c[row, col, frameNumber-buffer_size:frameNumber] + spatial_smoothness_weight*numpy.dot(gaussianWeightList, solutionArray)
                        currentSolution[:-1] = currentSolution[:-1] + temporal_smoothness_weight*differenceTrajectory[1:]
                        diagonalElements = 1 + spatial_smoothness_weight*numpy.dot(gaussianWeightList, numpy.ones((buffer_size,)))
                        diagonalElements[:-1] = diagonalElements[:-1] + temporal_smoothness_weight #last diagonal element
                        solutionArray = numpy.divide(currentSolution, diagonalElements)

                differenceTrajectory = numpy.asarray(solutionArray)
                y.append(solutionArray[-1])
                
            optimizedTrajectory[row, col, :] = numpy.asarray(y)


    return optimizedTrajectory

def capture_video(cap):
    """
    Reads the video and generates motion meshes and vertex profiles.

    Parameters:
        cap: cv2.VideoCapture object

    Returns:
        x_motion_meshes: motion meshes in x-direction
        y_motion_meshes: motion meshes in y-direction
        x_paths: vertex profiles in x-direction
        y_paths: vertex profiles in y-direction
    """

    global HORIZONTAL_BORDER
    HORIZONTAL_BORDER = 30

    # Take first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # preserve aspect ratio
    global VERTICAL_BORDER
    VERTICAL_BORDER = (HORIZONTAL_BORDER*old_gray.shape[1])//old_gray.shape[0]

    # motion meshes
    y_motion_meshes = []
    x_motion_meshes = []
    
    # paths
    y_paths = numpy.zeros((meshSize, meshSize, 1))
    x_paths = numpy.zeros((meshSize, meshSize, 1))
    
    Lucas_Kanade = dict( 
                        winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
                    )
    ShiTomasi = dict(  
                            maxCorners = 1000,
                            qualityLevel = 0.2,
                            minDistance = 7,
                            blockSize = 7 
                        )
    frame_num = 1
    while frame_num < total_frames:

        # processing frames
        _, frame = cap.read()
        print(frame.shape)
        #visualise mesh
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        for i in range(0, image.width, image.width//meshSize):
            draw.line((i, 0, i, image.height), fill=(255, 0, 0))
        for j in range(0, image.height, image.height//meshSize):
            draw.line((0, j, image.width, j), fill=(255, 0, 0))
        image.save(f"./output/image/mesh_{str(frame_num)}.png")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find corners in old frame to track in new frame
        prev_points = cv2.goodFeaturesToTrack(old_gray, mask=None, **ShiTomasi)

        # calculate optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray, prev_points, None, **Lucas_Kanade)

        # Select good points
        good_old = prev_points[status==1]
        good_new = next_points[status==1]

        # generate motion meshes
        x_motion_mesh, y_motion_mesh = transfer_motion(good_old, good_new, frame)
        if(x_motion_mesh == []):
            frame_num +=1
            continue

        try:
            y_motion_meshes = numpy.concatenate((y_motion_meshes, numpy.expand_dims(y_motion_mesh, axis=2)), axis=2)
            x_motion_meshes = numpy.concatenate((x_motion_meshes, numpy.expand_dims(x_motion_mesh, axis=2)), axis=2)
        except:
            y_motion_meshes = numpy.expand_dims(y_motion_mesh, axis=2)
            x_motion_meshes = numpy.expand_dims(x_motion_mesh, axis=2)

        # update vertex profiles
        x_paths, y_paths = update_profile(x_paths, y_paths, x_motion_mesh, y_motion_mesh)

        # while updation
        frame_num += 1

        old_frame = frame.copy()
        old_gray = gray.copy()


    return [x_motion_meshes, y_motion_meshes, x_paths, y_paths]


def get_frame_warp(x_motion_meshes, y_motion_meshes, x_paths, y_paths, sx_paths, sy_paths):
    """
    Generates frame warping for each frame.

    Parameters:
        x_motion_meshes: motion meshes in x-direction
        y_motion_meshes: motion meshes in y-direction
        x_paths: vertex profiles in x-direction
        y_paths: vertex profiles in y-direction
        sx_paths: optimized vertex profiles in x-direction
        sy_paths: optimized vertex profiles in y-direction

    Returns:
        frame_warp: frame warping for each frame
    """

    # U = P-C
    x_motion_meshes = numpy.concatenate((x_motion_meshes, numpy.expand_dims(x_motion_meshes[:, :, -1], axis=2)), axis=2)
    y_motion_meshes = numpy.concatenate((y_motion_meshes, numpy.expand_dims(y_motion_meshes[:, :, -1], axis=2)), axis=2)
    new_x_motion_meshes = sx_paths-x_paths
    new_y_motion_meshes = sy_paths-y_paths

    return x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes



def render_video(cap, x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes, COMPARE, MOTION_VECTORS):
    """
    Generates stabilized video, comparison video and saves them to the video/out directory.

    Parameters:
        cap: cv2.VideoCapture object that is instantiated with given video
        x_motion_meshes: motion meshes in x-direction
        y_motion_meshes: motion meshes in y-direction
        new_x_motion_meshes: updated motion meshes in x-direction
        new_y_motion_meshes: updated motion meshes in y-direction
        COMPARE: boolean to geberate a comparison of stabilized video with original video
        MOTION_VECTORS: boolean to geberate to visualize motion vectors

    Returns:
        None
    """

    old_rows = y_motion_meshes.shape[0]
    old_cols = y_motion_meshes.shape[1]
    new_rows = new_y_motion_meshes.shape[0]
    new_cols = new_y_motion_meshes.shape[1]
    print(old_rows, old_cols, new_rows, new_cols)
    # extract video parameters
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    rowWidth = height // meshSize
    colWidth = width // meshSize
    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # stabilized video path
    out = cv2.VideoWriter('./video/out/stablised_video.avi', fourcc, frame_rate, (width, height))
    
    if COMPARE:
        out_compare = cv2.VideoWriter('./video/out/compare.avi', fourcc, frame_rate, (2*width, height))

    frame_num = 0
    
    bar = tqdm(total=total_frames)
    while frame_num < total_frames:
        
            # reconstruct from frames
            _, frame = cap.read()

            x_motion_mesh = x_motion_meshes[:, :, frame_num]
            y_motion_mesh = y_motion_meshes[:, :, frame_num]
            new_x_motion_mesh = new_x_motion_meshes[:, :, frame_num]
            new_y_motion_mesh = new_y_motion_meshes[:, :, frame_num]
            
            # stabilize frame
            new_frame = frame_warp(frame, new_x_motion_mesh, new_y_motion_mesh)
            print("new frame : ", new_frame.shape)
            new_frame = new_frame[HORIZONTAL_BORDER:-HORIZONTAL_BORDER, VERTICAL_BORDER:-VERTICAL_BORDER, :]
            new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
            output = new_frame
            
            # write stabilized frame
            out.write(output)

            if COMPARE:
                output_compare = numpy.concatenate((frame, new_frame), axis=1)
                out_compare.write(output_compare)
            
            # visualize motion vectors
            if MOTION_VECTORS:
                # old motion vectors
                r = (((frame.shape[0]/meshSize)**2 + (frame.shape[1]/meshSize)**2)**0.5)/4
                for i in range(old_rows):
                    for j in range(old_cols):
                        theta = numpy.arctan2(y_motion_mesh[i, j], x_motion_mesh[i, j])
                        cv2.line(frame, (j*colWidth, i*rowWidth), (int(j*colWidth+r*numpy.cos(theta)), int(i*rowWidth+r*numpy.sin(theta))), 1)
                
                cv2.imwrite(f'./visual/old_vectors/{str(frame_num)}.jpg', frame)

                # new motion vectors
                for i in range(new_rows):
                    for j in range(new_cols):
                        theta = numpy.arctan2(new_y_motion_mesh[i, j], new_x_motion_mesh[i, j])
                        cv2.line(new_frame, (j*colWidth, i*rowWidth), (int(j*colWidth+r*numpy.cos(theta)), int(i*rowWidth+r*numpy.sin(theta))), 1)
                
                cv2.imwrite(f'./visual/new_vectors/{str(frame_num)}.jpg', new_frame)

            frame_num += 1
            bar.update(1)
        
    
    bar.close()

    # release video writer and file
    cap.release()
    out.release()


if __name__ == '__main__':
    
    start_time = time()
    
    # get video properties
    file_name = sys.argv[1]
    cap = cv2.VideoCapture(file_name)
    
    # propogate motion vectors and generate vertex profiles
    print(f'Currently Doing: Generating Vertex Profile')
    x_motion_meshes, y_motion_meshes, x_paths, y_paths = capture_video(cap)
    
    # stabilize the vertex profiles
    print("Currently Doing: Predictive Adaptive Path Smoothing")
    sy_paths = optimize_trajectory(y_paths)
    sx_paths = optimize_trajectory(x_paths)
    
    if PLOT:
        print("Optional Step: Plotting the vertex profiles")
        # visualize optimized paths
        plot_vertex_profiles(x_paths, sx_paths)
        print("Done plotting the vertex profiles")

    # get updated mesh warps
    x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes = get_frame_warp(x_motion_meshes, y_motion_meshes, x_paths, y_paths, sx_paths, sy_paths)

    # apply updated mesh warps & save the result
    print("Currently Doing: Generating the stabilized video")
    render_video(cap, x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes, COMPARE, MOTION_VECTORS)
    print((f'Total time taken: {str(time()-start_time)}s\n'))

    print("Done generating the stabilized video. Check the ./video/out folder for the result.\n")
