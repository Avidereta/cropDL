from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, recall_score, precision_score
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle
import numpy as np


class NoThresholdSpecified(Exception):
    def __init__(self, *args):
        self.message = """Threshold is not specified. Please either
            1) pass it as a parameter or
            2) run estimate_f1_max method to compute the threshold corresponded to maximum f1 score.
            """
        super(NoThresholdSpecified, self).__init__(self.message, *args)

class MetricsVisualizer:
    """
    Class for computing, visualizing and saving metrics

    Main params:
        model      -- prediction model
        generator  -- data generator
        n_batches -- number of batches


    Use cases:

        model = ...
        model2 = ...

        vis = MetricsVisualizer(model,test_generator,100)
        vis2 = MetricsVisualizer(model2,test_generator,100)

        vis.plot_ROC()
        vis2.plot_ROC()
        plt.show()


        print vis.estimate_...()
        print vis2.estimate_...()

    """

    def __init__(self,
                 model,
                 generator,
                 n_batches,
                 name=None,
                 info=False,
                 info_bin_threshold=0.5,
                 info_threshold=0.05
                 ):
        """

        :param model: prediction model
        :param generator: data generator
        :param n_batches: int, number of batches
        :param name: str, name of the model or experiment
        :param info: bool, informative content of visualized maps:
                if True then maps with lesions are selected
        :param info_bin_threshold: float, binarization for probabilities prediction:
                if a > info_bin_threshold, a is a lesion
        :param info_threshold: float in [0,1], percent of map's cloudy points to be informative
        """

        self.model = model
        self.name = name or self.model.name or ""
        self.gen = generator
        self.n_batches = n_batches

        self.info = info
        self.info_bin_threshold = info_bin_threshold
        self.info_threshold = info_threshold

        self.f1_score, self.threshold = None, None
        self.tpr, self.fpr = None, None
        self.auc, self.acc, self.recall, self.precision = None, None, None, None

        # generate set for metrics computing
        logs = []
        for i in range(self.n_batches):
            x_batch, y_batch, _ = self.gen.next()

            if info:
                info_percent = self._compute_info_percent(y_batch)
                while info_percent > info_threshold:
                    x_batch, y_batch, _ = self.gen.next()
                    info_percent = self._compute_info_percent(y_batch)

            y_class_predicted = model.predict(x_batch).ravel()
            y_true = y_batch.ravel()
            logs.append([y_true, y_class_predicted])

        self.y_true, self.y_class_predicted = map(np.concatenate, zip(*logs))

    def _compute_info_percent(self, batch):
        return np.sum(self.info_bin_threshold < batch) * 1. / np.size(batch.ravel())

    @staticmethod
    def compute_f1(tp, fp, tn, fn):
        """
        Computes f1 score for certain params
        :param tp: int, true positive
        :param fp: int, false positive
        :param tn: int, true negative
        :param fn: int, false negative
        :return: float, f1 score
        """
        recall = tp * 1. / (tp + fn)
        precision = tp * 1. / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def estimate_f1(self, threshold=None):

        """
        Computes f score for specified threshold or, if threshold is None, maximum f score for
        set of thresholds. Set of thresholds is generated from  probabilities in prediction matrix

        :param threshold: None or float, threshold for f score computing
        :return: float, f score corresponded to threshold (optimal of specified)
        """

        if threshold is None:

            thresholds_sorted = np.sort(self.y_class_predicted.ravel())
            y_true_sorted = self.y_true.ravel()[np.argsort(self.y_class_predicted.ravel())]

            cur_threshold = 0
            opt_threshold = 0
            tp = sum(i > 0 for i in y_true_sorted)
            fp = sum(i == 0 for i in y_true_sorted)
            tn = 0
            fn = 0
            f1_max = self.compute_f1(tp, fp, tn, fn)

            for i, y_i in enumerate(y_true_sorted[:-1]):
                if y_i:
                    tp -= 1
                    fn += 1
                    new_f1 = self.compute_f1(tp, fp, tn, fn)
                    if new_f1 > f1_max:
                        f1_max = new_f1
                        opt_threshold = cur_threshold

                else:
                    fp -= 1
                    tn += 1

                    if thresholds_sorted[i] != thresholds_sorted[i + 1]:
                        cur_threshold = (thresholds_sorted[i] + thresholds_sorted[i + 1]) * 0.5
                        new_f1 = self.compute_f1(tp, fp, tn, fn)
                        if new_f1 > f1_max:
                            f1_max = new_f1
                            opt_threshold = cur_threshold

            if not y_true_sorted[-1]:
                fp -= 1
                tn += 1
                new_f1 = self.compute_f1(tp, fp, tn, fn)
                if new_f1 > f1_max:
                    f1_max = new_f1
                    opt_threshold = cur_threshold

            self.threshold = opt_threshold
            self.f1_score = f1_max

        else:
            f1 = f1_score(self.y_true.ravel(), self.y_class_predicted.ravel() > threshold)
            self.threshold = threshold
            self.f1_score = f1

        return self.f1_score, self.threshold

    def estimate_auc(self):
        """:returns: a single float: classifier AUC"""

        auc = roc_auc_score(self.y_true.ravel(), self.y_class_predicted.ravel())
        return auc

    def plot_ROC(self,
                 name="",
                 label=None,
                 axistitles=True,
                 legend=True,
                 legend_loc='best',
                 grid=False,
                 include_fscore=False):
        """

        :param name:
        :param label: str, plot label
        :param axistitles: bool,
        :param legend: bool,
        :param legend_loc: location of legend
        :param grid: bool, True -- plot grid
        :param include_fscore: bool, True -- put computed f1 score on the plot
        :return:

        The location codes are:
        'best'         : 0, (only implemented for axes legends)
        'upper right'  : 1,
        'upper left'   : 2,
        'lower left'   : 3,
        'lower right'  : 4,
        'right'        : 5,
        'center left'  : 6,
        'center right' : 7,
        'lower center' : 8,
        'upper center' : 9,
        'center'       : 10,
        """

        self.fpr, self.tpr, _ = roc_curve(self.y_true.ravel(), self.y_class_predicted.ravel())

        if include_fscore:
            if self.f1_score is None:
                raise ValueError("f1 score is None. Firstly run class.estimate_f1_max with parameters you want")
            else:
                label = (label or ("ROC: " + self.name + ", f_score = " + str(round(self.f1_score, 3))))
        else:
            label = (label or ("ROC: " + self.name))

        plt.plot(self.fpr, self.tpr, label=label)
        title = "Model " + name + ", n_batches " + str(self.n_batches)
        plt.title(title)
        if axistitles:
            plt.xlabel('fpr')
            plt.ylabel('tpr')
        if legend:
            plt.legend(loc=legend_loc)
        if grid:
            plt.grid()

    def plot_maps(self,
                  max_ticks=None,
                  n_hidden=6,
                  info=False,
                  info_bin_threshold=1e-5,
                  info_threshold=0.05):
        """

        :param max_ticks: int, number of predictions to visualize
        :param n_hidden: int, number of hidden states to visualize,
              if n_hidden > max number o states, then all are visualized
        :param info: bool, informativity of visualized maps:
                if True then maps with clouds are selected
        :param info_bin_threshold: float, binarization for probabilities prediction:
                if a > info_bin_threshold, a is cloudy
        :param info_threshold: float in [0,1], percent of map's cloudy points to be informative
        :return:
        """

        pass

    def estimate_accuracy(self, threshold):
        """:return: a single float: classifier accuracy"""

        if threshold is None:
            if self.threshold is None:
                raise NoThresholdSpecified
            else:
                threshold = self.threshold

        self.acc = accuracy_score(self.y_true.ravel(), self.y_class_predicted.ravel() >= threshold)
        return self.acc

    def estimate_precision(self, threshold=None):
        """:return: a single float: classifier prediction"""

        if threshold is None:
            if self.threshold is None:
                raise NoThresholdSpecified
            else:
                threshold = self.threshold

        self.precision = precision_score(self.y_true.ravel(), self.y_class_predicted.ravel() >= threshold)
        return self.precision

    def estimate_recall(self, threshold=None):
        """:return: a single float: classifier recall"""

        if threshold is None:
            if self.threshold is None:
                raise NoThresholdSpecified
            else:
                threshold = self.threshold

        self.recall = recall_score(self.y_true.ravel(), self.y_class_predicted.ravel() >= threshold)
        return self.recall

    def save_results(self, filename):
        """
        Computes all scores if they are still None and saves them into the file
        :param filename: path to save
        :return:
        """

        if self.f1_score is None:
            raise ValueError("f1 score is None. Firstly run class.estimate_f1_max with parameters you want")

        if self.threshold is None:
            raise ValueError("threshold is None. Firstly run class.estimate_f1_max with parameters you want")

        if self.auc is None:
            self.estimate_auc()

        if any(self.tpr is None, self.fpr is None):
            self.fpr, self.tpr, _ = roc_curve(self.y_true.ravel(), self.y_class_predicted.ravel())

        if self.acc is None:
            self.estimate_accuracy(self.threshold)

        if self.precision is None:
            self.estimate_precision(self.threshold)

        if self.recall is None:
            self.estimate_recall(self.threshold)

        results = {str(self.name): OrderedDict([
            ("f1 score", self.f1_score),
            ("threshold", self.threshold),
            ("auc", self.auc),
            ("tpr", self.tpr),
            ("fpr", self.fpr),
            ("acc", self.acc),
            ("precision", self.precision),
            ("recall", self.recall),
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