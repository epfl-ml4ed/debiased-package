import numpy as np
import copy
import sklearn.gaussian_process as gp
from scipy.linalg import cholesky, cho_solve, solve_triangular
import warnings
from mitigation.postprocessing.nguyen_repo import group_fairness
from mitigation.postprocessing.nguyen_repo import individual_fairness

# compute mean function directly based on noise variance
def predict_w_noise(model_gp_, X_test_, noise_gp_):
    # model_gp_ is GP built on X_test_
    # X_test_ is validation/testing set
    # noise_gp_ noise used in GP

    # 1. compute mean function
    K = model_gp_.kernel_(model_gp_.X_train_)
    K[np.diag_indices_from(K)] += noise_gp_
    try:
        L_ = cholesky(K, lower=True)
        # self.L_ changed, self._K_inv needs to be recomputed
        _K_inv = None
    except np.linalg.LinAlgError as exc:
        exc.args = ("The kernel, %s, is not returning a "
                    "positive definite matrix. Try gradually "
                    "increasing the 'noise' parameter of your "
                    "GaussianProcessRegressor estimator."
                    % model_gp_.kernel_,) + exc.args
        raise
    dual_coef = cho_solve((L_, True), model_gp_.y_train_)
    # compute mean function
    K_trans = model_gp_.kernel_(X_test_, model_gp_.X_train_)
    y_mean = K_trans.dot(dual_coef)
    # undo normalization
    y_mean = model_gp_._y_train_std * y_mean + model_gp_._y_train_mean

    # 2. compute variance function
    # cache result of K_inv computation
    if _K_inv is None:
        # compute inverse K_inv of K based on its Cholesky
        # decomposition L and its inverse L_inv
        L_inv = solve_triangular(L_.T, np.eye(L_.shape[0]))
        _K_inv = L_inv.dot(L_inv.T)
    # compute variance of predictive distribution
    y_var = model_gp_.kernel_.diag(X_test_)
    y_var -= np.einsum("ij,ij->i", np.dot(K_trans, _K_inv), K_trans)
    # check if any of the variances is negative because of numerical issues. if yes: set the variance to 0.
    y_var_negative = y_var < 0
    if np.any(y_var_negative):
        warnings.warn("Predicted variances smaller than 0. "
                      "Setting those variances to 0.")
        y_var[y_var_negative] = 0.0
    # undo normalization
    y_var = y_var * model_gp_._y_train_std ** 2

    return y_mean, np.sqrt(y_var)

# objective function to optimize noise variance in GP (i.e. we consider predicted score of initial model is noisy)
def objective_function(
    features, model, y_pred_validation, 
    ground_truths, protected,
    noises_to_optimize, noise_gp_lower
):
    global acc_bo, fair_bo, theil_bo
    global relabel_bo
    global cnt_obj_func_optimize
    noises_to_optimize = np.array(noises_to_optimize).reshape(-1, 1)  # format noises_to_optimize to [[]]
    res = []
    relabel_bo = []
    for noise_to_optimize in noises_to_optimize:
        noise_gp = noise_to_optimize[0]
        # fix error in predict_w_noise() when noise_gp is negative
        if noise_gp < 0:
            noise_gp = noise_gp_lower
        print("noise_gp in objective function: {}".format(round(noise_gp, 4)))
        # compute the difference between relabeling function (i.e. mean function of GP) and initial function
        # on validation set but ONLY considering samples whose labels are changed
        ini_pred_validation = copy.deepcopy(y_pred_validation)
        ini_pred_validation_round = np.around(ini_pred_validation)
        gen_pred_validation, gen_std_validation = predict_w_noise(model, features, noise_gp)
        gen_pred_validation_round = np.around(gen_pred_validation)
        # reformat gen_pred_validation_round as same as y_valid
        gen_pred_validation_round = np.array(gen_pred_validation_round).reshape(-1, 1)
        # find samples whose labels are changed
        diff_indices = (ini_pred_validation_round != gen_pred_validation_round)
        if any(diff_indices):
            # relabeling function changes labels of some samples
            # we compute difference between initial function and relabeling function on these samples
            # in terms of predicted labels, NOT predicted scores
            difference_valid = len(np.where(diff_indices == True)[0]) / len(y_pred_validation)
        else:
            # relabel does not change labels of any samples
            # we penalize this relabeling function to set a max difference so that we will not choose it
            difference_valid = 1
        # compute accuracy and fairness of relabeling function w.r.t the sensitive feature on validation set
        accuracy_overall_valid, demographic_parity_valid, \
        prob_favored_pred_positive_valid, prob_unfavored_pred_positive_valid \
            = group_fairness.compute_accuracy_fairness(features, protected, ground_truths, gen_pred_validation_round)
        print("relabeling func on validation")
        print("difference={}, accuracy={}, fairness={}".
              format(round(difference_valid, 2), round(accuracy_overall_valid, 2), round(demographic_parity_valid, 2)))
        theil_index_valid = individual_fairness.generalized_entropy_index(ground_truths, gen_pred_validation_round)
        print("theil_index={}".format(round(theil_index_valid, 2)))
        # save new predicted labels on validation set of relabeling function
        relabel_bo.append(gen_pred_validation_round.reshape(1, -1)[0]) # need to reshape to array to add to array
        # compute score for BO (i.e. maximize both fairness and (1 - difference) between two functions)
        similarity_valid = (1 - difference_valid)
        score_bo = (1 - 0.5) * demographic_parity_valid + 0.5 * similarity_valid
        print("similarity: {}, fairness: {}, score_bo: {}".
              format(round(similarity_valid, 2), round(demographic_parity_valid, 2), round(score_bo, 2)))
        res.append([score_bo])
        cnt_obj_func_optimize = cnt_obj_func_optimize + 1
    res = np.array(res)

    return res[0], relabel_bo[0]
