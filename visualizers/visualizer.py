from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, \
    roc_curve, recall_score
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle

class Visualizer:
    """
    Class for computing, visualizing and saving metrics

    Main params:
        model      -- prediction model
        generator  -- data generator
        n_batches -- number of batches


    Use cases:

        model = ...
        model2 = ...

        vis = Visualizer(model,test_generator,100)
        vis2 = Visualizer(model2,test_generator,100)

        vis.plot_ROC()
        vis2.plot_ROC()
        plt.show()


        print vis.esimate_...()
        print vis2.estimate_...()

    """

    def __init__(self,
                 model,
                 generator,
                 n_batches,
                 name=None,
                    ):
        """

        :param model: prediction model
        :param generator: data generator
        :param n_batches: int, number of batches
        :param name: str, name of the model or experiment
        """

        self.model = model
        self.name = name or self.model.name or ""
        self.gen = generator
        self.n_batches = n_batches

        self.mse = None
        self.mae = None

        logs = []
        # generate set for metrics computing
        for i in range(self.n_batches):
            x_batch, y_batch = self.gen.next()
            y_pred = self.model.predict(x_batch).ravel()

            # set of y_true and y_predicted
            logs.append([y_batch, y_pred])

        self.y_true, self.y_pred = map(np.concatenate, zip(*logs))

    def estimate_mse(self):
        return ((self.y_true - self.y_pred)**2).mean()

    def estimate_mae(self):
        return (np.abs(self.y_true - self.y_pred)).mean()

    def plot_maps(self, n_hidden=8):
        """

        :param n_hidden: int, number of hidden states to visualize,
              if n_hidden > max number o states, then all are visualized
        """

        # compute

        x_seq, y_true = self.gen.next()
        y_pred = self.model.predict(x_seq, )[0]

        hid_maps = self.model.predict_hidden(x_seq)[0:n_hidden]

        x_seq = x_seq[0]

        # plot
        for i, prediction in enumerate(y_pred):

            width = 3
            height = 1 + np.int(np.ceil(n_hidden / (width * 1.)))

            fig, axs = plt.subplots(height, width, figsize=(30, 20))
            # fig.subplots_adjust(hspace=.1, wspace=.001)

            axs = axs.ravel()

            axs[0].imshow(x_seq[i], vmin=0, vmax=100)
            axs[0].set_title('X input. Y_true = {}, Y_pred = {}'.format(y_true[i], prediction))

            for n in range(n_hidden):
                axs[1 + n].imshow(hid_maps[n][i])
                axs[1 + n].set_title('hidden state %d' % (n + 1))


    def save_results(self, filename):
        """
        Computes all scores if they are still None and saves them into the file
        :param filename: path to save
        :return:
        """

        if self.mae is None:
            self.estimate_mae()
            # raise ValueError("MAE is None. Firstly run class.estimate_mae with parameters you want")

        if self.mse is None:
            self.estimate_mse()
            # raise ValueError("MSE is None. Firstly run class.estimate_mse with parameters you want")

        results = {str(self.name): OrderedDict([
            ("mae", self.mae),
            ("mse", self.mse),
            ("n batches", self.n_batches)])}

        # if file is not empty - extract and write new results
        try:
            with open(filename, "r") as f:
                results_in = pickle.load(f)
                results.update(results_in)

        except:
            pass

        # if file is empty - write new results
        with open(filename, "w") as f_out:
            pickle.dump(results, f_out, protocol=2)

        if self.name == "":
            print """Warning: There is no model name added to results,
            add it in class parameters and resave results."""