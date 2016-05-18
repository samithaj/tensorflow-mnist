import tensorflow as tf
import input_data
import cv2
import numpy as np
import math
from scipy import ndimage
import sys
import os

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)
    print cy,cx

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


"""
a placeholder for our image data:
None stands for an unspecified number of images
784 = 28*28 pixel
"""
x = tf.placeholder(tf.float32, [None, 784])
# we need our weights for our neural net
W = tf.Variable(tf.zeros([784,10]))
# and the biases
b = tf.Variable(tf.zeros([10]))

"""
softmax provides a probability based output
we need to multiply the image values x and the weights
and add the biases
(the normal procedure, explained in previous articles)
"""
y = tf.nn.softmax(tf.matmul(x,W) + b)

"""
y_ will be filled with the real values
which we want to train (digits 0-9)
for an undefined number of images
"""

y_ = tf.placeholder(tf.float32, [None, 10])
"""
conv layers
"""
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#con lay endddddd

"""
we use the cross_entropy function
which we want to minimize to improve our model
"""
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))


"""
use a learning rate of 0.01
to minimize the cross_entropy error
"""
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


image = sys.argv[1]
train = False if len(sys.argv) == 2 else sys.argv[2]
checkpoint_dir = "cps/"

saver = tf.train.Saver()
sess = tf.InteractiveSession()
# initialize all variables and run init
sess.run(tf.initialize_all_variables())
if train:
    # create a MNIST_data folder with the MNIST dataset if necessary
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # use 1000 batches with a size of 100 each to train our net
    # for i in range(1000):
    #     batch_xs, batch_ys = mnist.train.next_batch(100)
    #     # run the train_step function with the given image values (x) and the real output (y_)
    #     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for i in range(20000):
      batch = mnist.train.next_batch(50)        
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


    saver.save(sess, checkpoint_dir+'model.ckpt')
    """
    Let's get the accuracy of our model:
    our model is correct if the index with the highest y value
    is the same as in the real digit vector
    The mean of the correct_prediction gives us the accuracy.
    We need to run the accuracy function
    with our test set (mnist.test)
    We use the keys "images" and "labels" for x and y_
    """
    #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    
    
    # print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

else:
    # Here's where you're restoring the variables w and b.
    # Note that the graph is exactly as it was when the variables were
    # saved in a prior training run.
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print 'No checkpoint found'
        exit(1)



if not os.path.exists("img/" + image + ".png"):
    print "File img/" + image + ".png doesn't exist"
    exit(1)

# read original image
color_complete = cv2.imread("img/" + image + ".png")
height, width = color_complete.shape[:2]
color_complete = cv2.resize(color_complete,(5*width, 5*height), interpolation = cv2.INTER_CUBIC)

# read the bw image
gray_complete = cv2.imread("img/" + image + ".png", 0)
eight, width = gray_complete.shape[:2]
gray_complete = cv2.resize(gray_complete,(5*width, 5*height), interpolation = cv2.INTER_CUBIC)
gray_complete = (255-gray_complete)

# better black and white version
(thresh, gray_complete) = cv2.threshold(255-gray_complete, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("pro-img/compl.png", gray_complete)

digit_image = -np.ones(gray_complete.shape)

height, width = gray_complete.shape

"""
crop into several images
"""
for cropped_width in range(100, 300, 20):
    for cropped_height in range(100, 300, 20):
        for shift_x in range(0, width-cropped_width, cropped_width/4):
            for shift_y in range(0, height-cropped_height, cropped_height/4):
                gray = gray_complete[shift_y:shift_y+cropped_height,shift_x:shift_x + cropped_width]
                if np.count_nonzero(gray) <= 20:
                     continue

                if (np.sum(gray[0]) != 0) or (np.sum(gray[:,0]) != 0) or (np.sum(gray[-1]) != 0) or (np.sum(gray[:,
                                                                                                            -1]) != 0):
                    continue

                top_left = np.array([shift_y, shift_x])
                bottom_right = np.array([shift_y+cropped_height, shift_x + cropped_width])

                while np.sum(gray[0]) == 0:
                    top_left[0] += 1
                    gray = gray[1:]

                while np.sum(gray[:,0]) == 0:
                    top_left[1] += 1
                    gray = np.delete(gray,0,1)

                while np.sum(gray[-1]) == 0:
                    bottom_right[0] -= 1
                    gray = gray[:-1]

                while np.sum(gray[:,-1]) == 0:
                    bottom_right[1] -= 1
                    gray = np.delete(gray,-1,1)

                actual_w_h = bottom_right-top_left
                if (np.count_nonzero(digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]+1) >
                            0.2*actual_w_h[0]*actual_w_h[1]):
                    continue

                print "------------------"
                print "------------------"

                rows,cols = gray.shape
                compl_dif = abs(rows-cols)
                half_Sm = compl_dif/2
                half_Big = half_Sm if half_Sm*2 == compl_dif else half_Sm+1
                if rows > cols:
                    gray = np.lib.pad(gray,((0,0),(half_Sm,half_Big)),'constant')
                else:
                    gray = np.lib.pad(gray,((half_Sm,half_Big),(0,0)),'constant')

                gray = cv2.resize(gray, (20, 20))
                gray = np.lib.pad(gray,((4,4),(4,4)),'constant')


                shiftx,shifty = getBestShift(gray)
                shifted = shift(gray,shiftx,shifty)
                gray = shifted

                cv2.imwrite("pro-img/"+image+"_"+str(shift_x)+"_"+str(shift_y)+".png", gray)

                """
                all images in the training set have an range from 0-1
                and not from 0-255 so we divide our flatten images
                (a one dimensional vector with our 784 pixels)
                to use the same 0-1 based range
                """
                flatten = gray.flatten() / 255.0


                print "Prediction for ",(shift_x, shift_y, cropped_width)
                print "Pos"
                print top_left
                print bottom_right
                print actual_w_h
                print " "
                # prediction = [tf.reduce_max(y),tf.argmax(y,1)[0]]
                prediction=tf.argmax(y_conv,1)
                # pred = sess.run(prediction, feed_dict={x: [flatten]})
                pred = prediction.eval(feed_dict={x: [flatten],keep_prob: 1.0}, session=sess)
                print pred


                digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]] = pred[0]

                cv2.rectangle(color_complete,tuple(top_left[::-1]),tuple(bottom_right[::-1]),color=(0,255,0),thickness=5)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(color_complete,str(pred[0]),(top_left[1],bottom_right[0]+50),
                            font,fontScale=1.4,color=(0,255,0),thickness=4)
                cv2.putText(color_complete,format(pred[0]*100,".1f")+"%",(top_left[1]+30,bottom_right[0]+60),
                            font,fontScale=0.8,color=(0,255,0),thickness=2)



cv2.imwrite("pro-img/"+image+"_digitized_image.png", color_complete)
cv2.imshow("gg", color_complete)
cv2.waitKey()