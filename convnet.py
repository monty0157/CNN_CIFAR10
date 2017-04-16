import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import layers
import parameters as params
#import image_import as images
from data_import import unpickle

unpickled_data = unpickle('./data/cifar-10-batches-py/data_batch_2')

dataset = unpickled_data[b'data'].reshape(len(unpickled_data[b'data']),32,32,3)
labels = unpickled_data[b'labels']

#convert labels to one-hot vector
vector_labels = tf.one_hot(labels, depth=10)

n_classes = 10
training_epochs = 2
batch_size = 50

x_input = tf.placeholder(tf.float32, shape=(None,32,32,3), name="x-input")
y_output = tf.placeholder('float')

keep_prob = tf.placeholder('float')

def convolutional_neural_network(data):
    #data = tf.reshape(data, [-1, 160, 96, 3])

    #NAMING CONVENTION: conv3s32n is a convolutional layer with filter size 3x3 and number of filters = 32
    conv3s32n = layers.conv_layer(data, params.weights(depth=3), params.biases())
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(), params.biases())
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(), params.biases())


    #NAMING CONVENTION: pool2w2s is a pool layer with 2x2 window size and stride = 2
    pool2w2s = layers.pool_layer(conv3s32n)

    conv3s32n = layers.conv_layer(pool2w2s, params.weights(), params.biases())
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(), params.biases())
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(), params.biases())

    pool2w2s = layers.pool_layer(conv3s32n)

    conv3s32n = layers.conv_layer(pool2w2s, params.weights(n_filters=64), params.biases(n_filters=64))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=64, n_filters=64), params.biases(n_filters=64))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=64, n_filters=64), params.biases(n_filters=64))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=64, n_filters=64), params.biases(n_filters=64))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=64, n_filters=64), params.biases(n_filters=64))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=64, n_filters=64), params.biases(n_filters=64))

    pool2w2s = layers.pool_layer(conv3s32n)

    conv3s32n = layers.conv_layer(pool2w2s, params.weights(depth=64, n_filters=128), params.biases(n_filters=128))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=128, n_filters=128), params.biases(n_filters=128))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=128, n_filters=128), params.biases(n_filters=128))


    pool2w2s = layers.pool_layer(conv3s32n)

    conv3s32n = layers.conv_layer(pool2w2s, params.weights(depth=128, n_filters=256), params.biases(n_filters=256))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=256, n_filters=256), params.biases(n_filters=256))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=256, n_filters=256), params.biases(n_filters=256))

    #NAMING CONVENTION: Fully connected layers are just indexed
    fc1 = layers.full_layer(conv3s32n, params.fc_weights(conv3s32n, 1024), params.biases(1024), keep_prob)
    fc2 = layers.full_layer(fc1, params.fc_weights(fc1, 1024), params.biases(1024), keep_prob)
    fc3 = layers.full_layer(fc2, params.fc_weights(fc2, 1024), params.biases(1024), keep_prob)


    output = layers.output_layer(fc3, params.fc_weights(fc3, n_classes), params.biases(n_classes))

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y_output))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        tf.summary.scalar("cost", cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_output, 1))


    with tf.name_scope('accuracy'):
       accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
       tf.summary.scalar("accuracy", accuracy)



    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        summary_op = tf.summary.merge_all()
        #writer = tf.summary.FileWriter("./logs/saved_model_5", graph=tf.get_default_graph())

        sess.run(tf.global_variables_initializer())

        #FOR TRAINING THE NETWORK
        for epoch in range(training_epochs):
            epoch_loss = 0
            start = 0
            end = int(batch_size)
            for j in range(batch_size):
                epoch_x, epoch_y = dataset[start:end], vector_labels.eval()[start:end]
                j, c, summary = sess.run([optimizer, cost, summary_op], feed_dict = {x_input: epoch_x, y_output: epoch_y, keep_prob: 1.0 })
                epoch_loss += c
                start += int(batch_size)
                end += int(batch_size)
                print(epoch_loss)
                #writer.add_summary(summary, j)

            print('Epoch', epoch, 'completed out of', 10, 'loss:', epoch_loss, 'Accuracy:', accuracy.eval(feed_dict={x:epoch_x, y_output: epoch_y, keep_prob: 1.0}),)

            #save_path = saver.save(sess, "my-model")

        #model_import = tf.train.import_meta_graph('my-model.meta')
        #model_import.restore(sess, tf.train.latest_checkpoint('./'))

        '''print("TEST")
        total_accuracy = 0
        start = 0
        end = int(batch_size)
        for j in range(int(n_images_test_pos/batch_size)):
            acc_x, acc_y = image_import_test_pos[start:end], labels_test[start:end]
            total_accuracy += accuracy.eval(feed_dict={x:acc_x, y_output:acc_y, keep_prob: 1.0})
            start += int(batch_size+1)
            end += int(batch_size)

        mean_accuracy = total_accuracy/int(n_images_test_pos/batch_size)
        print(mean_accuracy)'''

        '''#FOR TESTING THE NETWORK
        model_import = tf.train.import_meta_graph('my-model.meta')
        model_import.restore(sess, tf.train.latest_checkpoint('./'))

        result = sess.run(tf.argmax(prediction,1), feed_dict={x: [image_import_train[1]], keep_prob: 1})

        print (' '.join(map(str, result)))

        plt.imshow(images.image_import_train()[1])
        plt.show()'''



train_neural_network(x_input)
