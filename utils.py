from keras import backend as K
import warnings
warnings.filterwarnings("ignore")

# Letters present in the Label Text
letters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def encode_words_labels(word):
    """
    Encodes the Ground Truth Labels to a list of Values like eg.HAT returns [17,10,29]
    """
    label_lst = []
    for char in word:
        # keeping 0 for blank and for padding labels
        label_lst.append(letters.find(char))
    return label_lst


def words_from_labels(labels):
    """
    converts the list of encoded integer labels to word strings like eg. [12,10,29] returns CAT 
    """
    txt = []
    for ele in labels:
        if ele == len(letters):  # CTC blank space
            txt.append("")
        else:
            # print(letters[ele])
            txt.append(letters[ele])
    return "".join(txt)


def ctc_loss_function(args):
    """
    CTC loss function takes the values passed from the model returns the CTC loss using Keras Backend ctc_batch_cost function
    """
    y_pred, y_true, input_length, label_length = args
    # since the first couple outputs of the RNN tend to be garbage we need to discard them, found this from other CRNN approaches
    # I Tried by including these outputs but the results turned out to be very bad and got very low accuracies on prediction
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
