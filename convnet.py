import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import layers
import parameters as params
#import image_import as images
from data_import import unpickle

n_classes = 10
n_images = 50000
n_images_test = 10000
training_epochs = 10
batch_size = 100

unpickled_data = unpickle('./data/train_data/')
unpickled_data_test = unpickle('./data/test_data/')

dataset, labels = unpickled_data
dataset_test, labels_test = unpickled_data_test

#improving image quality by transposing image instead of reshaping
dataset = dataset.reshape(n_images,3,32,32).transpose(0,2,3,1)
dataset_test = dataset_test.reshape(n_images_test,3,32,32).transpose(0,2,3,1)

#convert labels to one-hot vector
vector_labels = tf.one_hot(labels, depth=10)
vector_labels_test = tf.one_hot(labels_test, depth=10)

x_input = tf.placeholder(tf.float32, shape=(None,32,32,3), name="x-input")
y_output = tf.placeholder('float')

keep_prob = tf.placeholder('float')

def convolutional_neural_network(data):
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

    conv3s32n = layers.conv_layer(pool2w2s, params.weights(n_filters=128), params.biases(n_filters=128))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=128, n_filters=128), params.biases(n_filters=128))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=128, n_filters=128), params.biases(n_filters=128))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=128, n_filters=128), params.biases(n_filters=128))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=128, n_filters=128), params.biases(n_filters=128))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=128, n_filters=128), params.biases(n_filters=128))

    '''conv3s32n = layers.conv_layer(pool2w2s, params.weights(n_filters=64), params.biases(n_filters=64))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=64, n_filters=64), params.biases(n_filters=64))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=64, n_filters=64), params.biases(n_filters=64))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=64, n_filters=64), params.biases(n_filters=64))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=64, n_filters=64), params.biases(n_filters=64))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=64, n_filters=64), params.biases(n_filters=64))

    pool2w2s = layers.pool_layer(conv3s32n)

    conv3s32n = layers.conv_layer(pool2w2s, params.weights(depth=64, n_filters=128), params.biases(n_filters=128))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=128, n_filters=128), params.biases(n_filters=128))
    conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=128, n_filters=128), params.biases(n_filters=128))'''


    #pool2w2s = layers.pool_layer(conv3s32n)

    #conv3s32n = layers.conv_layer(pool2w2s, params.weights(depth=128, n_filters=256), params.biases(n_filters=256))
    #conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=256, n_filters=256), params.biases(n_filters=256))
    #conv3s32n = layers.conv_layer(conv3s32n, params.weights(depth=256, n_filters=256), params.biases(n_filters=256))

    #NAMING CONVENTION: Fully connected layers are just indexed
    fc1 = layers.full_layer(conv3s32n, params.fc_weights(conv3s32n, 1024), params.biases(1024), keep_prob)
    fc2 = layers.full_layer(fc1, params.fc_weights(fc1, 1024), params.biases(1024), keep_prob)
    fc3 = layers.full_layer(fc2, params.fc_weights(fc2, 1024), params.biases(1024), keep_prob)


    output = layers.output_layer(fc1, params.fc_weights(fc1, n_classes), params.biases(n_classes))

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_output))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        tf.summary.scalar("cost", cost)
    print(prediction, y_output)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_output, 1))


    with tf.name_scope('accuracy'):
       accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
       tf.summary.scalar("accuracy", accuracy)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/100_epochs", graph=tf.get_default_graph())
        sess.run(tf.global_variables_initializer())
        labels = vector_labels.eval().reshape(n_images,10)

        #FOR TRAINING THE NETWORK
        for epoch in range(training_epochs):
            epoch_loss = 0
            start = 0
            end = int(batch_size)
            labels = vector_labels.eval().reshape(n_images,10)
            for j in range(int(n_images/batch_size)):
                epoch_x, epoch_y = dataset[start:end], labels[start:end]
                j, c, summary = sess.run([optimizer, cost, summary_op], feed_dict = {x_input: epoch_x, y_output: epoch_y, keep_prob: 1.0 })
                epoch_loss += c
                start += int(batch_size)
                end += int(batch_size)
                writer.add_summary(summary, j)

            print('Epoch', epoch + 1, 'completed out of', 10, 'loss:', epoch_loss, 'Accuracy:', accuracy.eval(feed_dict={x:epoch_x, y_output: epoch_y, keep_prob: 1.0}),)

            #save_path = saver.save(sess, "my-model")

        #model_import = tf.train.import_meta_graph('my-model.meta')
        #model_import.restore(sess, tf.train.latest_checkpoint('./'))

        print("TEST")
        total_accuracy = 0
        start = 0
        end = int(batch_size)
        labels_test = vector_labels_test.eval().reshape(n_images_test,10)
        for j in range(int(n_images_test/batch_size)):
            acc_x, acc_y = dataset_test[start:end], labels_test[start:end]
            total_accuracy += accuracy.eval(feed_dict={x:acc_x, y_output:acc_y, keep_prob: 1.0})
            start += int(batch_size)
            end += int(batch_size)

        mean_accuracy = total_accuracy/int(n_images_test/batch_size)
        print(mean_accuracy)

        #FOR TESTING THE NETWORK
        #model_import = tf.train.import_meta_graph('my-model.meta')
        #model_import.restore(sess, tf.train.latest_checkpoint('./'))

        #result = sess.run(tf.argmax(prediction,1), feed_dict={x: [dataset[1]], keep_prob: 1})

        #print (' '.join(map(str, result)))
        print(labels[30000])
        plt.imshow(dataset[30000])
        #plt.show()


train_neural_network(x_input)
