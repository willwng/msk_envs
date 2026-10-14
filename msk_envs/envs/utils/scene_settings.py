from dataclasses import dataclass


@dataclass
class SceneSettings:
    """ What the viewer/renderer should draw around the model. Saved with each recorded frame """
    lanes: bool = False
    """ whether to render lane lines """

    curve: bool = False
    """ whether to render lane curve """

    lane_width: float = 0.0
    """ the width of lane lines, only used when lanes=True """

    meter_markers: bool = False
    """ whether to render meter markers every 10m """

    axes: bool = False
    """ whether to render axes """
