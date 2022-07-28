import tensorflow as tf
    

class Base(tf.keras.Model):
    def __init__(self):
        super(Base, self).__init__()
        self.labels = None
        self.labels_mask = None

    def setLabels(self, labels, labels_mask):
        self.labels = labels
        self.labels_mask = labels_mask

    def maskedSoftmaxXEntropy(self, preds):
        """Softmax cross-entropy loss with masking"""
        assert self.labels is not None
        assert self.labels_mask is not None
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds,
                                                       labels=self.labels)
        mask = tf.cast(self.labels_mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def maskedAccuracy(self, preds):
        """Accuracy with masking"""
        assert self.labels is not None
        assert self.labels_mask is not None
        correct_prediction = tf.equal(tf.argmax(preds, 1),
                                      tf.argmax(self.labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(self.labels_mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)
