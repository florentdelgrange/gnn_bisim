# import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

from data.dataset import Dataset
import models.gcn
import models.gnn
from models.gnn_bisim import GNNBisim


def trainModel(model, dataset, num_epochs=500, debug=False):
    # Train the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    val_acc_list = []
    for epoch in range(num_epochs):
        # we open a GradientTape to record the operations run during
        # the forward pass of the input data through the model,
        # this enables automatic differentiation
        with tf.GradientTape() as tape:
            # run the forward pass
            _ = model(dataset.features, training=True)
            # compute the loss value
            train_loss = sum(model.losses)
            # and the accuracy
            train_acc = model.metrics[0].result()
        # use the GradientTape to automatically retrieve the gradients
        # of the trainable weights w.r.t. the losses
        grads = tape.gradient(train_loss, model.trainable_weights)
        # run one step of gradient descent
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # we also run a validation forward pass to see how we are doing w.r.t.
        # to the validation part of the data set
        model.setLabels(dataset.labels_val, dataset.val_mask)
        _ = model(dataset.features)
        val_loss = sum(model.losses)
        val_acc = model.metrics[0].result()
        if debug:
            print(f"Epoch {epoch + 1:04d} " +
                  f"train loss={float(train_loss):.5f}, " +
                  f"train acc={float(train_acc):.5f}; " +
                  f"valid loss={float(val_loss):.5f}, " +
                  f"valid acc={float(val_acc):.5f}")
        val_acc_list.append(val_acc)

        # we reset the labels to the training ones
        model.setLabels(dataset.labels_train, dataset.train_mask)
    if debug:
        print("=== Training finished ===")
    # to conclude, we run a test forward pass
    model.setLabels(dataset.labels_test, dataset.test_mask)
    _ = model(dataset.features)
    test_loss = sum(model.losses)
    test_acc = model.metrics[0].result()
    if debug:
        print(f"test loss={float(test_loss):.5f}, " +
              f"test acc={float(test_acc):.5f}")

    return (test_loss, test_acc, val_acc_list)


def set1(plot=False):
    # First set of tests: comparing GNNs with A + pI
    # We prepare a dictionary of p-GNNs to compare
    gnns = {"2layr-gcn": models.gcn.GCN2p,
            "2layr-gnn": models.gnn.GNN2pBias}
    for dsname in ["cora", "citeseer", "pubmed"]:
        ds = Dataset(dsname)
        for name, GNN in gnns.items():
            for p in range(1, 10, 2):
                p = p / 10.0
                # We instantiate a model
                model = GNN(in_dim=ds.features.shape[1],
                            out_dim=ds.labels_train.shape[1],
                            nonzero_feat_shape=ds.features.data.shape,
                            graph=ds.graph,
                            labels=ds.labels_train,
                            labels_mask=ds.train_mask,
                            id_factor=p)
                # Train the model
                (loss, acc, val_accs) = trainModel(model, ds)
                print(f"== Trained GNN {name}-{p} on dataset {dsname} ==")
                print(f"test loss={float(loss):.5f}, " +
                      f"test acc={float(acc):.5f}")
                if plot:
                    plt.plot(range(len(val_accs)),
                             val_accs, label=f"{name}-{p}")
        if plot:
            plt.legend(loc="lower right")
            # plt.title(f"\"{dsname}\" dataset")
            plt.xlabel("Training epochs")
            plt.ylabel("Validation accuracy")
            plt.savefig(f"{dsname}-pvalues.pdf", format="pdf")
            plt.clf()


def set2(plot=False):
    # Second set of graphs: comparing GCN architectures
    # We prepare a dictionary of GNNs to compare
    gnns = {"gcn": models.gcn.GCN,
            "2layr-p-gcn": models.gcn.GCN2p}
    for dsname in ["cora", "citeseer", "pubmed"]:
        ds = Dataset(dsname)
        for name, GNN in gnns.items():
            # We instantiate a model
            model = GNN(in_dim=ds.features.shape[1],
                        out_dim=ds.labels_train.shape[1],
                        nonzero_feat_shape=ds.features.data.shape,
                        graph=ds.graph,
                        labels=ds.labels_train,
                        labels_mask=ds.train_mask)
            # Train the model
            (loss, acc, val_accs) = trainModel(model, ds)
            print(f"== Trained GNN {name} on dataset {dsname} ==")
            print(f"test loss={float(loss):.5f}, " +
                  f"test acc={float(acc):.5f}")
            if plot:
                plt.plot(range(len(val_accs)),
                         val_accs, label=f"{name}")
        if plot:
            plt.legend(loc="lower right")
            # plt.title(f"\"{dsname}\" dataset")
            plt.xlabel("Training epochs")
            plt.ylabel("Validation accuracy")
            plt.savefig(f"{dsname}-gcns.pdf", format="pdf")
            plt.clf()


def set3(plot=False):
    # Third set of graphs: comparing GNN architectures
    # We prepare a dictionary of GNNs to compare
    gnns = {"2 layers, Arch. (A2)": models.gnn.GNN2Grohe,
            "2 layers, Arch. (A3)": models.gnn.GNN2pBias}
    for dsname in ["cora", "citeseer", "pubmed"]:
        ds = Dataset(dsname)
        for name, GNN in gnns.items():
            # We instantiate a gcn model
            gcn = GNN(in_dim=ds.features.shape[1],
                      out_dim=ds.labels_train.shape[1],
                      nonzero_feat_shape=ds.features.data.shape,
                      graph=ds.graph,
                      labels=ds.labels_train,
                      labels_mask=ds.train_mask)
            # Train the gcn
            (loss, acc, val_accs) = trainModel(gcn, ds)
            print(f"== Trained GNN {name} on dataset {dsname} ==")
            print(f"test loss={float(loss):.5f}, " +
                  f"test acc={float(acc):.5f}")
            if plot:
                plt.plot(range(len(val_accs)),
                         val_accs, label=f"{name}")
        if plot:
            plt.legend(loc="lower right")
            # plt.title(f"\"{dsname}\" dataset")
            plt.xlabel("Training epochs")
            plt.ylabel("Validation accuracy")
            plt.savefig(f"{dsname}-gnns.pdf", format="pdf")
            plt.clf()


def set4(plot=False):
    # Fourth set of graphs: comparing GNN architectures
    # with and without degree information in the input
    # We prepare a dictionary of GNNs to compare
    gnns = {"2-layer GNN (A1)": models.gnn.GNN2Bias}
    for dsname in ["cora", "citeseer", "pubmed"]:
        for add_deg in [False, True]:
            ds = Dataset(dsname, add_degree=add_deg)
            for name, GNN in gnns.items():
                # We instantiate a model
                model = GNN(in_dim=ds.features.shape[1],
                            out_dim=ds.labels_train.shape[1],
                            nonzero_feat_shape=ds.features.data.shape,
                            graph=ds.graph,
                            labels=ds.labels_train,
                            labels_mask=ds.train_mask)
                # Train the model
                (loss, acc, val_accs) = trainModel(model, ds)
                if add_deg:
                    name += " + deg. info."
                print(f"== Trained GNN {name} on dataset {dsname} ==")
                print(f"test loss={float(loss):.5f}, " +
                      f"test acc={float(acc):.5f}")
                if plot:
                    plt.plot(range(len(val_accs)),
                             val_accs, label=f"{name}")
        if plot:
            plt.legend(loc="lower right")
            # plt.title(f"\"{dsname}\" dataset")
            plt.xlabel("Training epochs")
            plt.ylabel("Validation accuracy")
            plt.savefig(f"{dsname}-gnn-deg.pdf", format="pdf")
            plt.clf()

def bisimulation_set():
    dsname = "cora"
    ds = Dataset(dsname, add_degree=False)
    model = GNNBisim(
        input_dim=ds.features.shape[1],
        output_dim=ds.labels_train.shape[1],
        graph=ds.graph,)
    model(ds.features)

if __name__ == "__main__":
    bisimulation_set()
    exit(0)
