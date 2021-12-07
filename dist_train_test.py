from tensorflow._api.v2 import distribute
from tensorflow.python.distribute.reduce_util import ReduceOp
from tfr_dataset import train_ds, test_ds
import tensorflow as tf
from model import DIEN
import tensorflow.keras.metrics as metrics
from configs import epochs
###############################
#tf.version==2.7.0#
###############################

if __name__ == "__main__":
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
    test_dist_ds = strategy.experimental_distribute_dataset(test_ds)
    with strategy.scope():
        model = DIEN()
        train_binary_acc = metrics.BinaryAccuracy()
        test_binary_acc = metrics.BinaryAccuracy()
        train_roc_auc = metrics.AUC(from_logits=True)
        test_roc_auc = metrics.AUC(from_logits=True)
        train_loss = metrics.Mean()
        test_loss = metrics.Mean()
        optimizer = tf.keras.optimizers.Adam(clipnorm = 5.0)

        def train_step(inputs):
            with tf.GradientTape() as tape:
                loss, logits, labels = model(inputs,training=True)
            gradients = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            y_pred = tf.round(tf.sigmoid(logits))
            train_binary_acc.update_state(y_true=labels,y_pred=y_pred)
            train_roc_auc.update_state(y_true=labels,y_pred=logits)
            train_loss.update_state(loss)
            return loss

        def test_step(inputs):
            loss, logits, labels = model(inputs,training=True)
            y_pred = tf.round(tf.sigmoid(logits))
            test_binary_acc.update_state(y_true=labels,y_pred=y_pred)
            test_roc_auc.update_state(y_true=labels,y_pred=logits)
            test_loss.update_state(loss)
            return loss
        @tf.function(experimental_relax_shapes=True)
        def dist_train_step(inputs):
            per_replica_losses = strategy.run(train_step,args=(inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN,per_replica_losses,axis=None)
        @tf.function(experimental_relax_shapes=True)
        def dist_test_step(inputs):
            per_replica_losses = strategy.run(test_step,args=(inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN,per_replica_losses,axis=None)
    
    for epoch in range(1,epochs+1):
        for step,inputs in enumerate(train_dist_ds,start = 1):
            loss = dist_train_step(inputs)
            if step % 100 == 0:
                to_write = f"Epoch:{epoch}\tStep:{step}\tTrain loss:{train_loss.result().numpy()}\tTrain_acc:{train_binary_acc.result().numpy()}\tTrain_roc_auc:{train_roc_auc.result().numpy()}\n"
                with open("./report.txt","a") as f:
                    f.write(to_write)
                print(to_write)

        print(f"Epoch:{epoch}\ttrain loss:{train_loss.result().numpy()}\ttrain_acc:{train_binary_acc.result().numpy()}\t"
            f"train_roc_auc:{train_roc_auc.result().numpy()}\t")

        train_binary_acc.reset_state()
        train_roc_auc.reset_state()
        train_loss.reset_state()
        
        for step,inputs in enumerate(test_dist_ds,start = 1):
            loss = dist_test_step(inputs)
        to_write = f"Epoch:{epoch}\tTest loss:{test_loss.result().numpy()}\tTest_acc:{test_binary_acc.result().numpy()}\tTest_roc_auc:{test_roc_auc.result().numpy()}\n"
        
        with open("./report.txt","a") as f:
            f.write("\n")
            f.write(to_write)
            f.write("\n")
        
        print(f"Epoch:{epoch}\ttest loss:{test_loss.result().numpy()}\ttest_acc:{test_binary_acc.result().numpy()}\t"
            f"test_roc_auc:{test_roc_auc.result().numpy()}\t")
        
        test_binary_acc.reset_state()
        test_roc_auc.reset_state()
        test_loss.reset_state()



