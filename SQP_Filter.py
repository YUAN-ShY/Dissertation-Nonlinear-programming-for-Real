import numpy as np

# calculate the total constraints violation
def eval_cviol(nlp, x):
    c = nlp.cons(x)
    h = 0.0

    # Violation of equality constraints
    h += np.sum(np.maximum(0, c-nlp.cu) + np.maximum(0, nlp.cl-c))

    # Violation of inequality constraints on variables
    h += np.sum(np.maximum(0, x-nlp.bu) + np.maximum(0, nlp.bl-x))

    return h


# check if the new point is a "better point"
def is_improvement(obj, cviol, filter_points):

    if not filter_points:
        return True
    
    # plot the filter process
    # display_fp(filter_points, [obj, cviol])

    for point in filter_points:
        if obj >= point[0] and cviol >= point[1]:
            return False
    
    return True


# plot the filtter points
def display_fp(filter_points, new_point):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for point in filter_points:
        px, py = point
        ax.plot(px, py, 'bo')  
        ax.plot([px, px], [py, py+3], 'b-', linewidth=1)  
        ax.plot([px, px+3], [py, py], 'b-', linewidth=1)  

    ax.plot(new_point[0], new_point[1], 'ro') 
    plt.grid()
    plt.show()


# update the filter points list
def filter_update(filter_points, new_point):
    # filter_points = [point for point in filter_points if not (new_point[0] <= point[0] and new_point[1] <= point[1])]
    filter_points.append(new_point)
    return filter_points


