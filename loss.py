import torch
import torch.nn.functional as F

########################### GausianMLR Losses #################################

sqrt_two = 2**0.5
eps = 1.0e-4


# The probability of a Gaussian variable being positive
def gaussian_variable_positive_probability(z_mean, z_std):
    return 0.5 * (1 - torch.erf(-z_mean / (z_std * sqrt_two)))


# GaussianMLR Classification Loss
def GaussianMLRClassification(z_mean, z_std, labels):

    N, K = labels.shape

    bigger_zero_prob = gaussian_variable_positive_probability(z_mean, z_std) + eps
    smaller_zero_prob = 1 - bigger_zero_prob + 2 * eps

    bigger_zero_loss = torch.sum(-torch.log(bigger_zero_prob[labels > 0]))
    smaller_zero_loss = torch.sum(-torch.log(smaller_zero_prob[labels == 0]))

    return (bigger_zero_loss + smaller_zero_loss) / (N * K)


# GaussianMLR Ranking Loss
def GaussianMLRRanking(z_mean, z_logvar, labels):

    N, K = labels.shape

    pair_map = torch.tensor([(i, j) for i in range(K - 1) for j in range(i + 1, K)])

    left_means = z_mean[:, pair_map[:, 0]]
    left_logvars = z_logvar[:, pair_map[:, 0]]

    right_means = z_mean[:, pair_map[:, 1]]
    right_logvars = z_logvar[:, pair_map[:, 1]]

    diff_mean = left_means - right_means
    diff_std = torch.sqrt(torch.exp(left_logvars) + torch.exp(right_logvars))

    bigger_prob = gaussian_variable_positive_probability(diff_mean, diff_std) + eps
    smaller_prob = 1 - bigger_prob + 2 * eps

    left_labels = labels[:, pair_map[:, 0]]
    right_labels = labels[:, pair_map[:, 1]]

    gt_bigger_map = left_labels > right_labels
    gt_smaller_map = left_labels < right_labels

    bigger_loss = torch.sum(-torch.log(bigger_prob[gt_bigger_map]))
    smaller_loss = torch.sum(-torch.log(smaller_prob[gt_smaller_map]))

    norm_coeff_pairs = torch.sum(gt_bigger_map) + torch.sum(gt_smaller_map)

    return (bigger_loss + smaller_loss) / norm_coeff_pairs


def GaussianMLR(z_mean, z_logvar, labels):

    z_std = torch.exp(z_logvar / 2)
    classification_loss = GaussianMLRClassification(z_mean, z_std, labels)
    ranking_loss = GaussianMLRRanking(z_mean, z_logvar, labels)

    return {
        "classification_loss": classification_loss,
        "ranking_loss": ranking_loss,
    }


def weak_GaussianMLR(z_mean, z_logvar, labels):

    labels[labels > 0] = 1

    z_std = torch.exp(z_logvar / 2)
    classification_loss = GaussianMLRClassification(z_mean, z_std, labels)
    ranking_loss = GaussianMLRRanking(z_mean, z_logvar, labels)

    return {
        "classification_loss": classification_loss,
        "ranking_loss": ranking_loss,
    }


###############################################################################

################################ CLR Losses ###################################


def weak_CLR(pair_logits, labels):

    labels[labels > 0] = 1

    N, K = labels.shape
    labels = labels.float()
    labels = torch.cat((labels, torch.ones(N, 1, device=labels.device) * 0.5), dim=1)
    K += 1

    pair_map = torch.tensor([(i, j) for i in range(K - 1) for j in range(i + 1, K)])
    bigger_map = (labels[:, pair_map[:, 0]] > labels[:, pair_map[:, 1]]).float()

    return F.binary_cross_entropy_with_logits(pair_logits, bigger_map)


def strong_CLR(pair_logits, labels):

    N, K = labels.shape
    labels = labels.float()
    labels = torch.cat((labels, torch.ones(N, 1, device=labels.device) * 0.5), dim=1)
    K += 1

    pair_map = torch.tensor([(i, j) for i in range(K - 1) for j in range(i + 1, K)])
    bigger_map = (labels[:, pair_map[:, 0]] > labels[:, pair_map[:, 1]]).float()

    return F.binary_cross_entropy_with_logits(pair_logits, bigger_map)


###############################################################################

################################# LSEP ########################################


def MultiThresholdLoss(scores, threshold, labels):

    binary_labels = (labels != 0).float()
    diff = scores - threshold

    return F.binary_cross_entropy(F.sigmoid(diff), binary_labels)


def weak_LSEP(scores, labels):

    N, K = labels.shape
    binary_labels = (labels != 0).float()

    pair_map = torch.tensor([(i, j) for i in range(K - 1) for j in range(i + 1, K)])

    left_labels = binary_labels[:, pair_map[:, 0]]
    right_labels = binary_labels[:, pair_map[:, 1]]
    neg_map = left_labels > right_labels
    zero_map = left_labels == right_labels

    left_scores = scores[:, pair_map[:, 0]]
    right_scores = scores[:, pair_map[:, 1]]
    diff_scores = left_scores - right_scores
    diff_scores[neg_map] *= -1
    diff_scores[zero_map] = -float("inf")

    exp_scores = torch.exp(diff_scores)
    instance_sum = torch.log(exp_scores.sum(1) + 1)

    return torch.mean(instance_sum)


def strong_LSEP(scores, labels):

    N, K = labels.shape

    pair_map = torch.tensor([(i, j) for i in range(K - 1) for j in range(i + 1, K)])

    left_labels = labels[:, pair_map[:, 0]]
    right_labels = labels[:, pair_map[:, 1]]
    neg_map = left_labels > right_labels
    zero_map = left_labels == right_labels

    left_scores = scores[:, pair_map[:, 0]]
    right_scores = scores[:, pair_map[:, 1]]
    diff_scores = left_scores - right_scores
    diff_scores[neg_map] *= -1
    diff_scores[zero_map] = -float("inf")

    exp_scores = torch.exp(diff_scores)
    instance_sum = torch.log(exp_scores.sum(1) + 1)

    return torch.mean(instance_sum)


###############################################################################
