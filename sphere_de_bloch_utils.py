import numpy as np
import plotly.graph_objs as go
from numpy import pi, sin, cos
import pickle as pkl


### Helper functions ####
def plot_vect(fig, qstate, color="red", title=None):
    """
    Plots a vector from origin to 'endpoint' in figure 'fig'.
    Params:
            fig: plotly figure object
            endpoint: list with the 3D coordinates [x, y, z]
            color: Color of the line to plot (str)
            title: Label to give to the line, will appear on the legend
    """
    bvect = qstate_to_Bvect(qstate)

    x, y, z = bvect
    # Add cone pointer
    cone_size = 0.5
    fig.add_trace(
        go.Cone(
            x=[x - 0.15 * x],
            y=[y - 0.15 * y],
            z=[z - 0.15 * z],
            u=[cone_size * x],
            v=[cone_size * y],
            w=[cone_size * z],
            showlegend=False,
            showscale=False,
            colorscale=[[0, color], [1, color]],
        )
    )

    x = np.array([0, x])
    y = np.array([0, y])
    z = np.array([0, z])
    fig.add_scatter3d(x=x, y=y, z=z, mode="lines", line_width=6, line_color=color, name=title)


def add_axes(fig):
    """
    Plots x, y, z axes in black in figure 'fig'
    Params:
            fig: plotly figure object
    """
    qstates = [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2) * 1j], [1, 0]]
    for i, label in enumerate(["x", "y", "z"]):
        # endpts = [0, 0, 0]
        # endpts[i] = 1
        qstate = qstates[i]
        plot_vect(fig, qstate, "black", title=label)


def plot_Bloch_sphere(show_fig=False):
    """
    Original Code from: https://community.plotly.com/t/adding-wireframe-around-a-sphere/37661
    by user empet
    Note: Mesh doesnt show x = 0 but x = -0.01...
    --------------------
    Plots the Bloch Sphere mesh in an interactive window
    Params:
            show_fig: Show figure if True

    Returns:
            fig: figure object where Bloch sphere was plotted
    """
    theta = np.linspace(0, 2 * pi, 120)
    phi = np.linspace(0, pi, 60)

    x = []
    y = []
    z = []
    for t in [theta[10 * k] for k in range(12)]:  # meridians:
        x.extend(list(cos(t) * sin(phi)) + [None])  # None is inserted to mark the end of a meridian line
        y.extend(list(sin(t) * sin(phi)) + [None])
        z.extend(list(cos(phi)) + [None])

    for s in [phi[6 * k] for k in range(10)]:  # parallels
        x.extend(list(cos(theta) * sin(s)) + [None])  # None is inserted to mark the end of a parallel line
        y.extend(list(sin(theta) * sin(s)) + [None])
        z.extend([cos(s)] * 120 + [None])

    fig = go.Figure()
    fig.add_scatter3d(
        x=x, y=y, z=z, mode="lines", line_width=3, line_color="rgb(10,10,10)", opacity=0.3, name="Bloch Sphere"
    )

    # Illustrate xyz axes in black
    add_axes(fig)

    if show_fig:
        fig.show()

    return fig


def correction_angle_0(bvect, tol=1e-5):
    """
    Applies correction to get true 0.0 value
    hardcoded tolerance of 1e-5
    Params:
            bvect: numpy array [x, y, z] representing the state vector or quantum vector [a, b]
    Returns:
            bvect: bvect or qstate where extremely small values has been set to zero
    """

    for i in range(len(bvect)):
        # Check if bvect is a quantum state with complex values
        if isinstance(bvect[i], complex):
            # If so, apply the correction on the real and imag part
            if np.real(bvect[i]) > -tol and np.real(bvect[i]) < tol:
                bvect[i] = 0 + np.imag(bvect[i]) * 1j

            if np.imag(bvect[i]) > -tol and np.imag(bvect[i]) < tol:
                bvect[i] = np.real(bvect[i]) + 0 * 1j

        # If not, it is a bvect with only real values
        else:
            if bvect[i] > -tol and bvect[i] < tol:
                bvect[i] = 0
    return bvect


def qstate_to_Bvect(qstate):
    """
    Takes a quantum state written in computational basis vector representation
    and returns the Bloch vector representation in cartesian coordinates
    Params:
        qstate: list or array of 2 elements representing [a, b] for qstate = a|0>+b|1>
    Return:
        bvect: list of cartesian coordinates [x, y, z]
    """
    tol = 0.1  # tolerance
    alpha, beta = qstate[0], qstate[1]

    # Check qstate is a valid quantum state: |norm|^2 = 1
    if alpha * np.conj(alpha) + beta * np.conj(beta) > (1 + tol) or alpha * np.conj(alpha) + beta * np.conj(beta) < (
        1 - tol
    ):
        print(f"norm: ", alpha * np.conj(alpha) + beta * np.conj(beta))
        raise Exception(f"qstate is not a valid quantum state")

    # Get the Bloch angles from the quantum states amplitudes
    theta = 2 * np.arccos(alpha)  # azimuthal angle
    phi = np.arctan2(np.imag(beta), np.real(beta))  # zenith angle

    # Transform the Bloch angles into cartesian coordinates
    bvect = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]

    # Apply correction to retrieve true 0.0 values
    bvect = correction_angle_0(bvect)

    return bvect  # x, y, z


def bvect_to_qstate(bvect):
    """
        Takes a Bloch vector written as a cartesian np.array [x, y, z]
    and returns the quantum vector representation in the computational basis
    Params:
        bvect: list of cartesian coordinates [x, y, z]
    Return:
        qstate: list or array of 2 elements representing [a, b] for qstate = a|0>+b|1>
    """
    x, y, z = bvect
    # theta = np.arccos(z)
    # phi = np.atan2(y, x)
    a = np.cos(np.arccos(z) / 2)
    b = np.sin(np.arccos(z) / 2) * np.e ** (np.arctan2(y, x) * 1j)

    qstate = [a, b]

    # TODO:
    """
    - Written with imag --> get rid of imag??? ok, it's imag after all
    - +i doest work get [0.7071067811865476, 0] instead of [0.7071067811865475, 0.7071067811865475j]
    - -i doest work get [0.7071067811865476, 0] instead of [0.7071067811865475, (-0-0.7071067811865475j)]
    --> fix angle 0 correction
    """
    qstate = correction_angle_0(qstate)

    return qstate


def get_qstates_dict():
    """
    Generate a dictionary with remarkable quantum states
    :return: dictionary
    """
    qstates = {
        "+": [1 / np.sqrt(2), 1 / np.sqrt(2)],
        "-": [1 / np.sqrt(2), -1 / np.sqrt(2)],
        "+i": [1 / np.sqrt(2), 1 / np.sqrt(2) * 1j],
        "-i": [1 / np.sqrt(2), -1 / np.sqrt(2) * 1j],
        "0": [1, 0],
        "1": [0, 1],
    }
    return qstates


def clear(exercice):
    """
    Removes all added traces to recover the exercice
    :exercice: a plotly figure object where the Bloch Sphere is plotted as well as
               the trace of a sequence of gates
    """
    exercice.data = tuple(x for x in exercice.data if x.name in ["Bloch Sphere", "Exercice", "x", "y", "z"])
    return None


##########################################
###     QUANTUM GATES
### XGate, YGate, ZGate, HGate, SGate, TGate
### Rotation gates are not readily accessible
##########################################


def RX(theta, vect):
    """
    RX applies a rotation with angle theta about the x-axis to vector vect
    :theta: angle in rads
    :vect: cartesian coordinates of vector endpoint (numpy array)
    :return: endpoint of the rotated vector
    """
    RX = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

    rotated = np.matmul(RX, vect)
    rotated = correction_angle_0(rotated)
    return rotated


def RY(theta, vect):
    """
    RY applies a rotation with angle theta about the y-axis to vector vect
    :theta: angle in rads
    :vect: cartesian coordinates of vector endpoint (numpy array)
    :return: endpoint of the rotated vector
    """
    RY = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

    rotated = np.matmul(RY, vect)
    rotated = correction_angle_0(rotated)
    return rotated


def RZ(theta, vect):
    """
    RZ applies a rotation with angle theta about the z-axis to vector vect
    :theta: angle in rads
    :vect: cartesian coordinates of vector endpoint (numpy array)
    :return: endpoint of the rotated vector
    """
    RZ = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    rotated = np.matmul(RZ, vect)
    rotated = correction_angle_0(rotated)
    return rotated


def Raxis(theta, axis, vect):
    """
    Applies a rotation of 'theta' radians to 'vect' with respect to an 'axis' of rotation
    :theta: angle of rotation (radians)
    :axis: unitary vector with cartesian coordinates [ux, uy, uz], must be unitary (ux**2 + uy**2 + uz**2 = 1)
    :vect: vector with cartesian coordinates [x, y, z] (numpy array)
    :return: rotated: coordinates of the rotated vector (numpy array)
    """
    tol = 0.1  # tolerance
    ux, uy, uz = axis

    # Check if axis is valid: |norm| = 1
    if ux**2 + uy**2 + uz**2 > (1 + tol) or ux ** 2 + uy**2 + uz**2 > (1 + tol):
        print(f"norm: ", ux * np.conj(ux) + uy * np.conj(uy) + uz * np.conj(uz))
        raise Exception(f"axis is not unitary")

    # According to 'Rotation matrix from axis and angle'
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    a = 1 - np.cos(theta)
    Raxis = np.array(
        [
            [ux**2 * a + np.cos(theta), ux * uy * a - uz * np.sin(theta), ux * uz * a + uy * np.sin(theta)],
            [ux * uy * a + uz * np.sin(theta), uy**2 * a + np.cos(theta), uy * uz * a - ux * np.sin(theta)],
            [ux * uz * a - uy * np.sin(theta), uy * uz * a + ux * np.sin(theta), uz**2 * a + np.cos(theta)],
        ]
    )

    rotated = np.matmul(Raxis, vect)
    rotated = correction_angle_0(rotated)

    return rotated


def plot_trajectory(fig, theta, rotationType, bvect, axis=None, num_points=20, color="red"):
    """
    Computes the trajectory of a Bloch vector on the Bloch sphere
    after application of a specified rotation gate and returns the rotated vector.
    :fig: figure object where the trajectory will be plotted
    :theta: angle of rotation in rads
    :rotationType: "RX", "RY", "RZ"
    :bvect: cartesian coordinates of a Bloch vector
    :num_points: number of points in the trajectory discretization between original bvect and rotated vector
    :line_color: color of the trajectory
    :return: cartesian coordinates of the rotated bvect
    """

    # Discretize the angles from 0 to theta
    angles = np.linspace(0, theta, num_points)

    # Apply specified rotation
    if rotationType == "RX":
        tmp_trace = RX(angles[0], bvect)
        for angle in angles[1:]:
            vtemp = RX(angle, bvect)
            tmp_trace = np.vstack((tmp_trace, vtemp))

    elif rotationType == "RY":
        tmp_trace = RY(angles[0], bvect)
        for angle in angles[1:]:
            vtemp = RY(angle, bvect)
            tmp_trace = np.vstack((tmp_trace, vtemp))

        bvect = RY(np.pi, bvect)

    elif rotationType == "RZ":
        tmp_trace = RZ(angles[0], bvect)
        for angle in angles[1:]:
            vtemp = RZ(angle, bvect)
            tmp_trace = np.vstack((tmp_trace, vtemp))

    elif rotationType == "Raxis":
        tmp_trace = Raxis(angles[0], axis, bvect)
        for angle in angles[1:]:

            vtemp = Raxis(angle, axis, bvect)
            tmp_trace = np.vstack((tmp_trace, vtemp))

    else:
        raise Exception(f"rotationtype not defined.")

    # Plot trajectory
    fig.add_scatter3d(
        x=tmp_trace[:, 0], y=tmp_trace[:, 1], z=tmp_trace[:, 2], mode="lines", line_width=5, line_color=color, opacity=1
    )

    # return rotated vector
    return tmp_trace[-1, :]


# def XGate(fig, bvect, debug=False, color='red'):
#     """
#     Applies the X gate to the bloch vector bvect
#     and plots the trajectory of the bloch vector during the rotation
#     :bvect: cartesian coordinates of a valid Bloch vector's endpoint
#     :return: endpoint of the rotated vector
#     """
#     # The X gate is equivalent to a rotation of pi about the x-axis on the Bloch sphere
#     bvect = plot_trajectory(fig, np.pi, "RX", bvect, color=color)

#     return bvect


def XGate(fig, qstate, debug=False, color="red"):
    """
    Applies the X gate to the bloch vector bvect
    and plots the trajectory of the bloch vector during the rotation
    :qstate: list or array of 2 elements representing [a, b] for qstate = a|0>+b|1>
    :return: endpoint of the rotated vector
    """
    bvect = qstate_to_Bvect(qstate)
    # The X gate is equivalent to a rotation of pi about the x-axis on the Bloch sphere
    bvect = plot_trajectory(fig, np.pi, "RX", bvect, color=color)
    new_qstate = bvect_to_qstate(bvect)
    return new_qstate


def YGate(fig, qstate, color="red"):
    """
    Applies the Y gate to the bloch vector bvect
    and plots the trajectory of the bloch vector during the rotation
    :qstate: list or array of 2 elements representing [a, b] for qstate = a|0>+b|1>
    :return: endpoint of the rotated vector

    Note: Y Gate is equivalent to a rotation of pi about the y-axis BUT there is a global phase difference of -i
    i.e: RY(pi) = -iY
    The global phase is not observable for states |0> and |1> since they are on the top/bottom of the sphere
    where phi's value can't be observed...
    TODO: confirm explanation and rationale
    """
    bvect = qstate_to_Bvect(qstate)
    bvect = plot_trajectory(fig, np.pi, "RY", bvect, color=color)
    new_qstate = bvect_to_qstate(bvect)
    return new_qstate


def ZGate(fig, qstate, color="red"):
    """
    Applies the Z gate to the bloch vector bvect
    and plots the trajectory of the bloch vector during the rotation
    :qstate: list or array of 2 elements representing [a, b] for qstate = a|0>+b|1>
    :return: endpoint of the rotated vector
    """
    bvect = qstate_to_Bvect(qstate)
    bvect = plot_trajectory(fig, np.pi, "RZ", bvect, color=color)
    new_qstate = bvect_to_qstate(bvect)
    return new_qstate


def SGate(fig, qstate, color="red"):
    """
    Applies the Z gate to the bloch vector bvect
    and plots the trajectory of the bloch vector during the rotation
    :qstate: list or array of 2 elements representing [a, b] for qstate = a|0>+b|1>
    :return: endpoint of the rotated vector
    """
    bvect = qstate_to_Bvect(qstate)
    bvect = plot_trajectory(fig, np.pi / 2, "RZ", bvect, color=color)
    new_qstate = bvect_to_qstate(bvect)
    return new_qstate


def TGate(fig, qstate, color="red"):
    """
    Applies the Z gate to the bloch vector bvect
    and plots the trajectory of the bloch vector during the rotation
    :qstate: list or array of 2 elements representing [a, b] for qstate = a|0>+b|1>
    :return: endpoint of the rotated vector
    """
    bvect = qstate_to_Bvect(qstate)
    bvect = plot_trajectory(fig, np.pi / 4, "RZ", bvect, color=color)
    new_qstate = bvect_to_qstate(bvect)
    return new_qstate


def HGate(fig, qstate, color="red"):
    """
    Applies the Hadamard gate to the bloch vector bvect
    and plots the trajectory of the bloch vector during the rotation
    :qstate: list or array of 2 elements representing [a, b] for qstate = a|0>+b|1>
    :return: endpoint of the rotated vector
    """
    bvect = qstate_to_Bvect(qstate)

    # Unit vector for x+z axis
    axis = [1 / np.sqrt(2), 0, 1 / np.sqrt(2)]

    # Hadamard is equivalent to a single rotation about the X+Z axis
    bvect = plot_trajectory(fig, np.pi, "Raxis", bvect, axis=axis, color=color)
    new_qstate = bvect_to_qstate(bvect)
    return new_qstate


def RZGate(fig, theta, qstate, color="red"):
    """
    Applies a rotation theta about the z-axis to the bloch vector bvect
    and plots the trajectory of the bloch vector during the rotation
    :fig: Figure object
    :theta: angle of rotation in rads
    :qstate: list or array of 2 elements representing [a, b] for qstate = a|0>+b|1>
    :return: endpoint of the rotated vector
    """
    bvect = qstate_to_Bvect(qstate)
    bvect = plot_trajectory(fig, theta, "RZ", bvect, color=color)
    new_qstate = bvect_to_qstate(bvect)
    return new_qstate


def RYGate(fig, theta, qstate, color="red"):
    """
    Applies a rotation theta about the y-axis to the bloch vector bvect
    and plots the trajectory of the bloch vector during the rotation
    :fig: Figure object
    :theta: angle of rotation in rads
    :qstate: list or array of 2 elements representing [a, b] for qstate = a|0>+b|1>
    :return: endpoint of the rotated vector
    """
    bvect = qstate_to_Bvect(qstate)
    bvect = plot_trajectory(fig, theta, "RY", bvect, color=color)
    new_qstate = bvect_to_qstate(bvect)
    return new_qstate


def RXGate(fig, theta, qstate, color="red"):
    """
    Applies a rotation theta about the x-axis to the bloch vector bvect
    and plots the trajectory of the bloch vector during the rotation
    :fig: Figure object
    :theta: angle of rotation in rads
    :qstate: list or array of 2 elements representing [a, b] for qstate = a|0>+b|1>
    :return: endpoint of the rotated vector
    """
    bvect = qstate_to_Bvect(qstate)
    bvect = plot_trajectory(fig, theta, "RX", bvect, color=color)
    new_qstate = bvect_to_qstate(bvect)
    return new_qstate


###############################################
####             EXERCICES                #####
###############################################


def load_exercices():
    exercices = pkl.load(open("exercices.pickle", "rb"))
    return exercices


def nothing():
    return None
