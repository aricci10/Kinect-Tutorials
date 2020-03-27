from pykinect2 import PyKinectV2 as Pk 
from pykinect2 import PyKinectRuntime
import numpy as np
import open3d as o3d
import time
import ctypes


class SampleListener:
    
    # define constants of the sizes of the frames
    COLOR_WIDTH = 1920 
    COLOR_HEIGHT = 1080

    DEPTH_WIDTH = 512
    DEPTH_HEIGHT = 424

    def __init__(self):

        self.done = False # <-- Kinect is not done yet

        #We want to ask for color frames and depth frames
        self.kinect = PyKinectRuntime.PyKinectRuntime( 
            Pk.FrameSourceTypes_Color | Pk.FrameSourceTypes_Depth)

        #create the buffers to store the mapped points.
        #buffer for the depth points mapped to RGB space
        depth2color_points_type = Pk._ColorSpacePoint * np.int(self.DEPTH_WIDTH * self.DEPTH_HEIGHT)
        self.depth2color_points = ctypes.cast(depth2color_points_type(), ctypes.POINTER(_ColorSpacePoint))

        #buffer for depth points mapped to camera (x,y,z) space
        depth2camera_points_type = Pk._CameraSpacePoint * np.int(self.DEPTH_WIDTH * self.DEPTH_HEIGHT)
        self.depth2camera_points = ctypes.cast(depth2camera_points_type(), ctypes.POINTER(_CameraSpacePoint))



    def generate_point_clouds(self, last_rgb_frame, last_depth_frame):

        #initial = time.time()
        self._kinect._mapper.MapDepthFrameToCameraSpace(ctypes.c_uint(self.DEPTH_WIDTH * self.DEPTH_HEIGHT),
                                                        last_depth_frame, #<-- use depth frame as input 
                                                        ctypes.c_uint(self.DEPTH_WIDTH * self.DEPTH_HEIGHT),
                                                        self.depth2camera_points) #<-- store in camera buffer
        self._kinect._mapper.MapDepthFrameToColorSpace(ctypes.c_uint(self.DEPTH_WIDTH * self.DEPTH_HEIGHT),
                                                       last_depth_frame, #<-- use depth frame as input
                                                       ctypes.c_uint(self.DEPTH_WIDTH * self.DEPTH_HEIGHT),
                                                       self.depth2color_points) #<-- store in color buffer

        #unpack the depth2camera_points C array into a numpy matrix with shape
        # (self.DEPTH_HEIGHT * self.DEPTH_WIDTH, 3) each row contains the x, y, z points
        xyz_array = np.ctypeslib.as_array(self.depth2camera_points, (self.DEPTH_HEIGHT * self.DEPTH_WIDTH,))
        xyz_array = xyz_array.view((xyz_array.dtype[0], 3))

        #unpack the depth2color_points C array into a numpy matrix with shape
        #(self.DEPTH_HEIGHT * self.DEPTH_WIDTH, 2) each row contains the the row and 
        #the column of the corresponding point in color space, for each point in depth space
        depth_xy_array = np.ctypeslib.as_array(self.depth2color_points, (self.DEPTH_HEIGHT * self.DEPTH_WIDTH,))
        depth_xy_array = depth_xy_array.view((depth_xy_array.dtype[0], 2))

        #sometimes the mapping function will throw inf and -inf, so we remove them
        xyz_array[xyz_array < -100000] = 0
        xyz_array[xyz_array > 100000] = 0

        #also we'll round the values of the row and columns so they can be used
        #as indexes. Notice the copy=false for efficiency
        x_values = depth_xy_array[:, 0].astype(int, copy=False)
        y_values = depth_xy_array[:, 1].astype(int, copy=False)

        #the output frame is flattened, so we have to create an absolute index
        idxes = 4 * (x_values + self.COLOR_WIDTH * y_values)

        #remove points which fall outside color space when registration is done
        idxes[idxes < 0] = 0
        idxes[idxes > self.COLOR_HEIGHT * self.COLOR_HEIGHT * 4] = 0

        #pixels are flattened in the B-R-G order
        b_array = last_rgb_frame[idxes].reshape((self.DEPTH_HEIGHT * self.DEPTH_WIDTH, 1))
        g_array = last_rgb_frame[idxes + 1].reshape((self.DEPTH_HEIGHT * self.DEPTH_WIDTH, 1))
        r_array = last_rgb_frame[idxes + 2].reshape((self.DEPTH_HEIGHT * self.DEPTH_WIDTH, 1))

        #create a unique array with the colors
        rgb_array = np.concatenate((b_array, g_array, r_array), axis=1) / 255
        
        #TODO: check this line
        rgb_array = rgb_array.astype(np.float16, copy=False)

        #remove points where values are [0,0,0] for xyz and rgb
        rgb_array = rgb_array[~np.all(xyz_array == 0, axis=1)]
        xyz_array = xyz_array[~np.all(xyz_array == 0, axis=1)]

        with open ("{}.npy".format(time.time()), 'w') as file:
            np.save(file, np.concatenate((xyz_array, rgb_array), axis = 1))

        #initial = time.time()
        #pcd = o3d.geometry.PointCloud()

        #pcd.points = o3d.utility.Vector3dVector(xyz_array.astype(np.float64))
        #pcd.colors = o3d.utility.Vector3dVector(rgb_array.astype(np.float64))
        #pcd = pcd.voxel_down_sample(voxel_size=0.02)
        #print(time.time() - initial)

        #o3d.io.write_point_cloud("{}.pcd".format(time.time()), pcd)

        # o3d.io.write_point_cloud("data/three.js-dev/examples/pcd/last.pcd", pcd)

    def run(self):
        # -------- Main Program Loop -----------

        while not self.done:

            #wait to have one of each frames
            if self._kinect.has_new_color_frame() and self._kinect.has_new_depth_frame():

                #get a color frame as a numpy array
                last_rgb_frame = self._kinect.get_last_color_frame()

                #get depth frame in original version to use in map function latter
                last_depth_frame = self._kinect._depth_frame_data 

                self.generate_point_clouds(last_rgb_frame, last_depth_frame)

        self._kinect.close()

if __name__ == "__main__":

    listener = SampleListener()
    listener.run()