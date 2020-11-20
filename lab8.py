# MIT 6.034 Lab 8: Support Vector Machines
# Written by 6.034 staff

from svm_data import *
from functools import reduce
import math

#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    """Computes the dot product of two vectors u and v, each represented 
    as a tuple or list of coordinates. Assume the two vectors are the
    same length."""
    return sum([i[0]*i[1] for i in list(zip(u,v))])

def norm(v):
    """Computes the norm (length) of a vector v, represented 
    as a tuple or list of coords."""
    return math.sqrt(dot_product(v,v))


#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
    w = svm.w
    b = svm.b
    x = point.coords
    return dot_product(w, x) + b

def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""
    value = positiveness(svm, point)
    
    if value == 0:
        return 0

    return (1 if value > 0 else -1)

def margin_width(svm):
    """Calculate margin width based on the current boundary."""
    w = svm.w
    norm_w = norm(w)
    return 2 / norm_w

def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    support_vectors = svm.support_vectors
    training_points = svm.training_points

    bad_points = set()
    
    # checking gutter constraint (positiveness == classification)
    for point in support_vectors:
        if positiveness(svm, point) != point.classification:
            bad_points.add(point)

    # check training points not between gutter 
    # positiveness will be less than 1 if not in the gutter
    for point in training_points:
        if abs(positiveness(svm, point)) < 1:
            bad_points.add(point)
        
    return bad_points


#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""
    support_vectors = svm.support_vectors
    training_points = svm.training_points

    bad_points = set()

    for point in training_points:
        # support vectors
        if point in support_vectors:
            if not point.alpha > 0:
                bad_points.add(point)
        # non support vectors
        else:
            if point.alpha != 0:
                bad_points.add(point)
        
    return bad_points

def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    training_points = svm.training_points
    w = svm.w

    # no training points
    if training_points == []:
        return True 

    # equation 4
    eq4 = sum(point.classification * point.alpha for point in training_points)
    if eq4 != 0:
        return False

    # equation 5
    eq5 = scalar_mult(training_points[0].classification * training_points[0].alpha, training_points[0].coords)
    for i in range(1, len(training_points)):
        point = training_points[i]
        eq5 = vector_add(eq5, scalar_mult(point.classification * point.alpha, point.coords))
    
    if eq5 != w:
        return False

    return True


#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    training_points = svm.training_points
    bad_points = set()

    for point in training_points:
        if point.classification != classify(svm, point):
            bad_points.add(point)
    
    return bad_points


#### Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""
    training_points = svm.training_points
    
    # no training points
    if training_points == []:
        return svm

    # calculating w
    w = scalar_mult(training_points[0].classification * training_points[0].alpha, training_points[0].coords)
    for i in range(1, len(training_points)):
        point = training_points[i]
        w = vector_add(w, scalar_mult(point.classification * point.alpha, point.coords))
    svm.w = w

    # setting the support vectors and b
    support_vectors = []
    b_negative_support = []
    b_positive_support = []
    for point in training_points:
        if point.alpha > 0:
            support_vectors.append(point)
            if point.classification < 0:
                b_negative_support.append(-1 - dot_product(w, point.coords))
            elif point.classification > 0:
                b_positive_support.append(1 - dot_product(w, point.coords))
    svm.b = (min(b_negative_support) + max(b_positive_support))/2
    svm.support_vectors = support_vectors

    return svm



#### Part 6: Multiple Choice ###################################################

ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = ['A', 'D']
ANSWER_6 = ['A', 'B', 'D']
ANSWER_7 = ['A', 'B', 'D']
ANSWER_8 = []
ANSWER_9 = ['A', 'B', 'D']
ANSWER_10 = ['A', 'B', 'D']

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1, 3, 6, 8]
ANSWER_18 = [1, 2, 4, 5, 6, 7, 8]
ANSWER_19 = [1, 2, 4, 5, 6, 7, 8]

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = "Nabib Ahmed"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = 6
WHAT_I_FOUND_INTERESTING = "The graphics"
WHAT_I_FOUND_BORING = "Nothing; fun lab"
SUGGESTIONS = "Great instructions!"
