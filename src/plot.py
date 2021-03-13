import copy

import numpy as np
import pandas as pd
import plotly.express as px


def plot3d(points, found_point=None, optimal_point=None, fname="figs/plot.html"):
    if len(points[0]) != 3:
        print("Not plotting")
        return
    new_points = [list(x) + ["agent"] for x in points]
    if found_point is not None:
        new_points.append(list(found_point) + ["found"])
    if optimal_point is not None:
        new_points.append(optimal_point + ["optimal"])

    new_points.append(list(np.mean(points, axis=0)) + ["mean"])
    df = pd.DataFrame(new_points, columns=["x", "y", "z", "color"])
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="color",
        range_x=[-0.1, 1.1],
        range_y=[-0.1, 1.1],
        range_z=[-0.1, 1.1],
        template="plotly_white",
    )
    print(f"Saved image to {fname}")
    fig.write_html(fname)
