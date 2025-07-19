class Dataset:
    """
    Parent class of noise filtering methods
    """
    def __init__(self, x_train, protected_attribute):
        """
        Constructor
        """
        self.data = x_train
        self.protected_attributes