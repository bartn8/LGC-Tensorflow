import tensorflow as tf
import time

import importlib
ops = importlib.import_module("LGC-Tensorflow.model.ops")
#import ops

#import numpy as np

class LGCBlock(object):

    def __init__(self, late_fusion, patch_size=9, model='LGC', verbose=False):                  
        self.sess = tf.compat.v1.Session()
        self.disposed = False
        self.patch_size = patch_size
        self.radius = int(patch_size/2)
        self.model = model
        self.late_fusion = late_fusion
        self.logName = "LGC Block"
        self.verbose = verbose

        self.build_model()

    def dispose(self):
        if not self.disposed:
            self.sess.close()
            self.disposed = True

    def log(self, x):
        if self.verbose:
            print(f"{self.logName}: {x}")

    def build_model(self):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log(f"Building Model ({self.model}) ...")

        self.left = tf.compat.v1.placeholder(tf.float32, name='left')
        self.disp = tf.compat.v1.placeholder(tf.float32, name='disparity')
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
        
        {'CCNN': self.EFN, 
         'EFN':  self.EFN,
         'LFN':  self.LFN, 
         'ConfNet': self.ConfNet,
         'LGC': self.LGC}[self.model]()

    def EFN(self): #CCNN/EFN
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log(" [*] Building EFN model...")

        kernel_size = 3
        filters = 64
        fc_filters = 100

        if self.model == "EFN":
            self.log(" [*] Building EFN model...")
            nchannels = 4
            model_input = tf.concat([self.disp, self.left], axis=3)
        else: #CCNN
            self.log(" [*] Building CCNN model...")
            nchannels=1
            if self.model == 'LGC':
                disp = self.disp
                model_input = tf.pad(disp, [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]])
            else:
                model_input = self.disp

        with tf.compat.v1.variable_scope('CCNN'):
            with tf.compat.v1.variable_scope("conv1"):
                conv1 = ops.conv2d(model_input, [kernel_size, kernel_size, nchannels, filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope("conv2"):
                conv2 = ops.conv2d(conv1, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope("conv3"):
                conv3 = ops.conv2d(conv2, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope("conv4"):
                conv4 = ops.conv2d(conv3, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope("fully_connected_1"):
                fc1 = ops.conv2d(conv4, [1, 1, filters, fc_filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope("fully_connected_2"):
                fc2 = ops.conv2d(fc1, [1, 1, fc_filters, fc_filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope("prediction"):
                if self.model == 'LGC':
                    self.local_prediction = tf.nn.sigmoid(ops.conv2d(fc2, [1, 1, fc_filters, 1], 1, False, padding='VALID'))
                else:
                    self.prediction = ops.conv2d(fc2, [1, 1, fc_filters, 1], 1, False, padding='VALID')

    def LFN(self):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log(" [*] Building LFN model...")

        kernel_size = 3
        filters = 64
        fc_filters = 100

        if self.model == 'LGC':
            disp, left = (self.disp, self.left)
            model_input_disp = tf.pad(disp, [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]])
            model_input_left = tf.pad(left, [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]])
        else:
            model_input_disp = self.disp
            model_input_left = self.left

        with tf.compat.v1.variable_scope('LFN'):
            with tf.compat.v1.variable_scope('disparity'):
                with tf.compat.v1.variable_scope("conv1"):
                    conv1_disp = ops.conv2d(model_input_disp, [kernel_size, kernel_size, 1, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv2"):
                    conv2_disp = ops.conv2d(conv1_disp, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv3"):
                    conv3_disp = ops.conv2d(conv2_disp, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv4"):
                    conv4_disp = ops.conv2d(conv3_disp, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope('RGB'):
                with tf.compat.v1.variable_scope("conv1"):
                    conv1_left = ops.conv2d(model_input_left, [kernel_size, kernel_size, 3, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv2"):
                    conv2_left = ops.conv2d(conv1_left, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv3"):
                    conv3_left = ops.conv2d(conv2_left, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv4"):
                    conv4_left = ops.conv2d(conv3_left, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope("fully_connected_1"):
                fc1 = ops.conv2d(tf.concat([conv4_left, conv4_disp], axis=3), [1, 1, 2 * filters, fc_filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope("fully_connected_2"):
                fc2 = ops.conv2d(fc1, [1, 1, fc_filters, fc_filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope("prediction"):
                if self.model == 'LGC':
                    self.local_prediction = tf.nn.sigmoid(ops.conv2d(fc2, [1, 1, fc_filters, 1], 1, False, padding='VALID'))
                else:
                    self.prediction = ops.conv2d(fc2, [1, 1, fc_filters, 1], 1, False, padding='VALID')

    def LGC(self):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log(" [*] Building LGC model...")

        kernel_size = 3
        filters = 64
        fc_filters = 100
        scale=255.0

        self.LFN() if self.late_fusion else self.EFN() 
        self.ConfNet()

        model_input_disp = self.disp
        model_input_local, model_input_global = (self.local_prediction, self.global_prediction)

        with tf.compat.v1.variable_scope('LGC'):
            with tf.compat.v1.variable_scope('disparity'):

                with tf.compat.v1.variable_scope("conv1"):
                    conv1_disp = ops.conv2d(model_input_disp, [kernel_size, kernel_size, 1, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv2"):
                    conv2_disp = ops.conv2d(conv1_disp, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv3"):
                    conv3_disp = ops.conv2d(conv2_disp, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv4"):
                    conv4_disp = ops.conv2d(conv3_disp, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope('local'):
                with tf.compat.v1.variable_scope("conv1"):
                    conv1_local = ops.conv2d(model_input_local*scale, [kernel_size, kernel_size, 1, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv2"):
                    conv2_local = ops.conv2d(conv1_local, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv3"):
                    conv3_local = ops.conv2d(conv2_local, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv4"):
                    conv4_local = ops.conv2d(conv3_local, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope('global'):
                with tf.compat.v1.variable_scope("conv1"):
                    conv1_global = ops.conv2d(model_input_global*scale, [kernel_size, kernel_size, 1, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv2"):
                    conv2_global = ops.conv2d(conv1_global, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv3"):
                    conv3_global = ops.conv2d(conv2_global, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.compat.v1.variable_scope("conv4"):
                    conv4_global = ops.conv2d(conv3_global, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope("fully_connected_1"):             
                fc1 = ops.conv2d(tf.concat([conv4_global, conv4_local, conv4_disp], axis=3), [1, 1, 3 * filters, fc_filters], 1, True, padding='VALID')

            with tf.compat.v1.variable_scope("fully_connected_2"):
                fc2 = ops.conv2d(fc1, [1, 1, fc_filters, fc_filters], 1, True, padding='VALID')
                
            with tf.compat.v1.variable_scope("prediction"):
                self.prediction = ops.conv2d(fc2, [1, 1, fc_filters, 1], 1, False, padding='VALID')

    def ConfNet(self):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log(" [*] Building ConfNet model...")

        kernel_size = 3
        filters = 32

        left = self.left
        disp = self.disp

        with tf.compat.v1.variable_scope('ConfNet'):
            with tf.compat.v1.variable_scope('RGB'):
                with tf.compat.v1.variable_scope("conv1"):
                    self.conv1_RGB = ops.conv2d(left, [kernel_size, kernel_size, 3, filters], 1, True, padding='SAME')
                                    
            with tf.compat.v1.variable_scope('disparity'):  
                with tf.compat.v1.variable_scope("conv1"):
                    self.conv1_disparity = ops.conv2d(disp, [kernel_size, kernel_size, 1, filters], 1, True, padding='SAME')
            
            model_input = tf.concat([self.conv1_RGB, self.conv1_disparity], axis=3)
            
            self.net1, self.scale1 = ops.encoding_unit('1', model_input, filters * 2)
            self.net2, self.scale2 = ops.encoding_unit('2', self.net1,   filters * 4)
            self.net3, self.scale3 = ops.encoding_unit('3', self.net2,   filters * 8)
            self.net4, self.scale4 = ops.encoding_unit('4', self.net3,   filters * 16)
            
            self.net5 = ops.decoding_unit('4', self.net4, num_outputs=filters * 8, forwards=self.scale4)
            self.net6 = ops.decoding_unit('3', self.net5, num_outputs=filters * 4, forwards=self.scale3)
            self.net7 = ops.decoding_unit('2', self.net6, num_outputs=filters * 2,  forwards=self.scale2)
            self.net8 = ops.decoding_unit('1', self.net7, num_outputs=filters, forwards=model_input)
                        
            if self.model == 'LGC':
                self.global_prediction = tf.nn.sigmoid(ops.conv2d(self.net8, [kernel_size, kernel_size, filters, 1], 1, False, padding='SAME'))
            else:
                self.prediction = ops.conv2d(self.net8, [kernel_size, kernel_size, filters, 1], 1, False, padding='SAME')

    def load(self, checkpoint_path):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())

        if self.model == 'LGC':
            self.vars = tf.all_variables()
            self.vars_global = [k for k in self.vars if k.name.startswith('ConfNet')]
            self.vars_local = [k for k in self.vars if (k.name.startswith('CCNN') or k.name.startswith('LFN')) ]
            self.vars_lgc = [k for k in self.vars if k.name.startswith('LGC')]

            self.saver_global = tf.compat.v1.train.Saver(self.vars_global)
            self.saver_local = tf.compat.v1.train.Saver(self.vars_local)
            self.saver_LGC = tf.compat.v1.train.Saver(self.vars_lgc)

            if checkpoint_path[0] and checkpoint_path[1] and checkpoint_path[2]:
                self.saver_global.restore(self.sess, checkpoint_path[0])
                self.saver_local.restore(self.sess, checkpoint_path[1])
                self.saver_LGC.restore(self.sess, checkpoint_path[2])
            else:
                self.log(" [*] Load failed...neglected")
                self.log(" [*] End Testing...")
                raise ValueError('checkpoint_path[0] or checkpoint_path[1] or checkpoint_path[2] is None')
        else:
            self.saver = tf.compat.v1.train.Saver()

            if checkpoint_path[0]:
                self.saver.restore(self.sess, checkpoint_path[0])
                self.log(" [*] Load model: SUCCESS")
            else:
                self.log(" [*] Load failed...neglected")
                self.log(" [*] End Testing...")
                raise ValueError('checkpoint_path is None')

    def test(self, left, disp):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log(" [*] Start Testing...")

        if self.model == 'ConfNet':
            prediction = tf.nn.sigmoid(self.prediction)
        else:
            prediction = tf.pad(tf.nn.sigmoid(self.prediction), tf.constant([[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]]), "CONSTANT")
        
        if self.model == 'ConfNet' or self.model == 'LGC':
            val_disp, hpad, wpad = ops.pad(disp)
            val_left, _, _ = ops.pad(left)

        start = time.time()
        if self.model == 'ConfNet' or self.model == 'LGC':
            confidence = self.sess.run(prediction, feed_dict={self.left: val_left, self.disp: val_disp})
            confidence = ops.depad(confidence, hpad, wpad)
        else:
            confidence = self.sess.run(prediction, feed_dict={self.left: left, self.disp: disp})
        current = time.time()

        #myConfidence = (confidence[0] * 255.0).astype('uint8')
        #myConfidence = (confidence[0] * 65535.0).astype('uint16')
        myConfidence = (confidence[0]).astype('float32')
        self.log(" [*] Inference time:" + str(current - start) + "s")

        return myConfidence
