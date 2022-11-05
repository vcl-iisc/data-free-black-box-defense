from torchmetrics import Accuracy


class Metric(Accuracy):
    def __init__(self, train=True):
        """
        :param train: set to True if metric is train time
        """

        super(Metric, self).__init__()

        self.clean_acc = Accuracy()
        self.wv_clean_acc = Accuracy()
        self.regen_clean_acc = Accuracy()

        self.adv_acc = Accuracy()
        self.wv_adv_acc = Accuracy()
        self.regen_adv_acc = Accuracy()
        self.key = "train" if train else "test"

    def update(self, predictions, labels):
        
        accuracy = {
                    "{}_clean_acc".format(self.key): self.clean_acc(predictions["pred_clean_images"], labels),
                    "{}_wv_clean_acc".format(self.key): self.wv_clean_acc(predictions["pred_clean_wv_images"], labels),
                    "{}_regen_clean_acc".format(self.key): self.regen_clean_acc(predictions["pred_clean_regen_images"],
                                                                                labels),
                    "{}_adv_acc".format(self.key): self.adv_acc(predictions["pred_adv_images"], labels),
                    "{}_wv_adv_acc".format(self.key): self.wv_adv_acc(predictions["pred_adv_wv_images"], labels),
                    "{}_regen_adv_acc".format(self.key): self.regen_adv_acc(predictions["pred_adv_regen_images"], labels)}

        return accuracy

    def compute(self):
        accuracy = {
            "{}_clean_acc".format(self.key): self.clean_acc.compute(),
                    "{}_wv_clean_acc".format(self.key): self.wv_clean_acc.compute(),
                    "{}_regen_clean_acc".format(self.key): self.regen_clean_acc.compute(),
                    "{}_adv_acc".format(self.key): self.adv_acc.compute(),
                    "{}_wv_adv_acc".format(self.key): self.wv_adv_acc.compute(),
                    "{}_regen_adv_acc".format(self.key): self.regen_adv_acc.compute()}

        return accuracy
