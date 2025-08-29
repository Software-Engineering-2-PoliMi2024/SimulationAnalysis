import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from typing import List, Tuple, Union
from scipy.interpolate import CubicSpline
import numpy as np
sns.set_theme()


def plotDistribution(data, title: str, xlabel: str, ylabel: str):
    """Plot the distribution of the given data with an histogram and kde.
    The mean and the std are also displayed.

    Args:
        data (np.ndarray): The input data to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data, bins=30, kde=True, ax=ax)
    ax.vlines(data.mean(), 0, 3, color='r', linestyle='--', label='$\\mu$')
    ax.vlines(data.mean() + data.std(), 0, 3, color='b',
              linestyle='--', label='$\\mu + \\sigma$')
    ax.vlines(data.mean() - data.std(), 0, 3, color='b',
              linestyle='--', label='$\\mu - \\sigma$')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()


def plotBivariateDistribution(x, y, title: str, xlabel: str, ylabel: str):
    """Plot the bivariate distribution of the given data with a scatter plot and marginal histograms with kde.

    Args:
        x (np.ndarray): The input data for the x-axis.
        y (np.ndarray): The input data for the y-axis.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    g = sns.jointplot(x=x, y=y, kind="scatter",
                      marginal_kws=dict(bins=30, fill=True, kde=True), s=20, alpha=0.5)
    g.figure.set_size_inches(8, 5)
    g.set_axis_labels(xlabel, ylabel)
    g.figure.suptitle(title)
    g.figure.tight_layout()
    g.figure.subplots_adjust(top=0.95)
    plt.show()


def plotRoad(controlPoints: List[Tuple[int, int, int, int]], ax: Axes, splineRenderPoints: int = 100, roadWidth: float = 3):
    """Plots a road given its control points.

    Args:
        controlPoints (List[Tuple[int, int, int, int]]): The control points of the road.
        ax (Axes): The axes to plot on.
        splineRenderPoints (int, optional): The number of points to use for rendering the spline. Defaults to 100.
        roadWidth (float, optional): The width of the plotted road. Defaults to 4.
    """
    xs = [p[0] for p in controlPoints]
    ys = [p[1] + roadWidth/4 for p in controlPoints]
    cs = CubicSpline(xs, ys)
    xs_fine = np.linspace(min(xs), max(xs), splineRenderPoints)
    # compute centerline and its derivative
    y = cs(xs_fine)
    dy = cs.derivative()(xs_fine)

    # unit normal (-dy, 1)/sqrt(1+dy^2)
    nx = -dy
    ny = 1.0
    denom = np.hypot(nx, ny)
    nx /= denom
    ny /= denom

    half = roadWidth / 2.0  # roadWidth is now in data (axis) units
    left_x = xs_fine - nx * half
    left_y = y - ny * half
    right_x = xs_fine + nx * half
    right_y = y + ny * half

    # polygon for the road area
    xs_poly = np.concatenate([left_x, right_x[::-1]])
    ys_poly = np.concatenate([left_y, right_y[::-1]])
    ax.fill(xs_poly, ys_poly, color='gray', zorder=1)

    # draw centerline on top
    ax.plot(xs_fine, y, color='white', linestyle='--')


def plotTrajectory(
        positions: List[Tuple[float, float]],
        ax: Axes, color: str | float | Tuple[float, float, float, float] = 'red',
        colorMap: Colormap | None = None,
        norm: Normalize | None = None,
        label: str | None = None,
        alpha: float = 1.0):
    """Plots a trajectory given its positions.

    Args:
        positions (List[Tuple[float, float]]): The positions of the trajectory.
        ax (Axes): The axes to plot on.
        color (str | float | Tuple[float, float, float, float], optional): The color of the trajectory. Defaults to 'red'.
        colorMap (Colormap | None, optional): The colormap to use for coloring the trajectory. Defaults to None.
        norm (Normalize | None, optional): The normalization to use for the trajectory color. Defaults to None.
        label (str | None, optional): The label for the trajectory. Defaults to None.
        alpha (float, optional): The alpha transparency of the trajectory. Defaults to 1.0.
    """

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    # If a colormap is provided and color is a scalar, map it to an RGBA tuple.
    color_arg = color
    if colorMap is not None:
        if norm is None:
            norm = Normalize()
        try:
            if np.isscalar(color):
                # map single scalar to RGBA using the provided colormap and normalization
                color_arg = colorMap(norm(float(color)))  # type: ignore
        except Exception:
            # fallback to the original color if mapping fails
            color_arg = color

    if label is not None:
        ax.plot(xs, ys, color=color_arg, label=label, alpha=alpha)
        ax.legend()
    else:
        ax.plot(xs, ys, color=color_arg, alpha=alpha)


def plotTrajectoriesOfARoad(
        roadControlPoints: List[Tuple[int, int, int, int]],
        trajectories: List[List[Tuple[float, float]]],
        ax: Axes | None = None,
        trajectoriesColor: str | float | Tuple[float, float, float,
                                               float] | List[Union[str, float, Tuple[float, float, float, float]]] = 'red',
        trajectoriesColorName: str | None = None,
        roadWidth: float = 3.0
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    plotRoad(roadControlPoints, ax, roadWidth=roadWidth)

    colorMap = plt.get_cmap("viridis")
    if isinstance(trajectoriesColor, list) and len(trajectoriesColor) == len(trajectories) and all(isinstance(c, (int, float)) for c in trajectoriesColor):
        norm = plt.Normalize(vmin=min(trajectoriesColor),  # type: ignore
                             vmax=max(trajectoriesColor))  # type: ignore
    else:
        norm = plt.Normalize(vmin=0, vmax=1)  # type: ignore

    for i, traj in enumerate(trajectories):
        plotTrajectory(
            positions=traj,
            ax=ax,
            color=trajectoriesColor[i] if isinstance(
                trajectoriesColor, list) else trajectoriesColor,
            colorMap=colorMap,
            norm=norm)

    ax.set_title('Simulation trajectories')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    if trajectoriesColorName is not None and isinstance(trajectoriesColor, list) and len(trajectoriesColor) == len(trajectories) and all(isinstance(c, (int, float)) for c in trajectoriesColor):
        # create a ScalarMappable for the colorbar using the same colormap and normalization
        sm = plt.cm.ScalarMappable(cmap=colorMap, norm=norm)
        sm.set_array([])  # required for colorbar to work with ScalarMappable
        plt.colorbar(sm, ax=ax, label=trajectoriesColorName)
