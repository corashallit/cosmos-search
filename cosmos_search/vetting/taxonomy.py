"""Explanation classes a vetter can assign, shared by the Gaia and TESS arms.

Verdicts are probability distributions over the leaves below. The three groups
are what the shortlist arithmetic needs (docs/vetting.md §5): a candidate stays
on the shortlist only when "unexplained" survives every stage. Leaves are kept
coarse on purpose: each one must be something a vetter can tell apart from a
packet, and something we can supply known-answer controls or decoys for.
"""

INSTRUMENTAL = {
    "blend_or_crowding": "Flux from a neighbour enters or leaves the window or aperture (scan angle, pointing, PSF)",
    "scattered_light": "Earth, Moon or other stray light raises the background; subtraction makes the star look faint",
    "pointing_or_momentum": "Pointing jitter, momentum dumps, attitude events, safe modes",
    "detector_defect": "Cosmic ray, hot or dead pixel or column, CCD edge, gating or saturation",
    "calibration_offset": "Offsets between fields of view, CCDs, cameras, sectors or calibration epochs",
    "shared_systematic": "Dimming shared with unrelated stars observed at the same time",
    "processing": "Pipeline artefact: detrending, background model, aperture choice, cross-match error",
    "other_instrumental": "Instrumental, none of the above",
}

ASTROPHYSICAL = {
    "eclipsing_binary": "Stellar eclipses, including grazing and long-period systems",
    "disk_occultation": "Long-duration occultation by a circumstellar or circumsecondary disk",
    "yso_dipper": "Young-star dipper or UX Ori variable: dust from an inner disk",
    "rcrb_dust": "R CrB-type dust formation: deep, reddened, irregular fades",
    "debris_transit": "Transiting dust or debris from a disintegrating body (planetesimals, comets, white-dwarf debris)",
    "planet_transit": "Planetary transit",
    "rotation_or_spots": "Rotational modulation by spots or faculae",
    "pulsation_or_irregular": "Pulsation, semi-regular or irregular variability of evolved stars",
    "eruptive_or_be": "Be-star, shell or other eruptive variability",
    "solar_system_occultation": "Occultation by, or confusion with, a solar-system body",
    "extragalactic": "The source is a galaxy or AGN",
    "other_astrophysical": "Astrophysical, none of the above",
}

UNEXPLAINED = {
    "unexplained": "No instrumental or known astrophysical explanation fits the data in the packet",
}

GROUPS = {"instrumental": INSTRUMENTAL, "astrophysical": ASTROPHYSICAL, "unexplained": UNEXPLAINED}
LEAVES = tuple(k for g in GROUPS.values() for k in g)
GROUP_OF = {leaf: g for g, members in GROUPS.items() for leaf in members}


def group_probs(class_probs):
    """Sum leaf probabilities into the three groups."""
    out = {g: 0.0 for g in GROUPS}
    for leaf, p in class_probs.items():
        out[GROUP_OF[leaf]] += float(p)
    return out


def describe():
    """The class list as text, for the expert prompt."""
    lines = []
    for g, members in GROUPS.items():
        lines.append(f"{g}:")
        lines.extend(f"  - {leaf}: {text}" for leaf, text in members.items())
    return "\n".join(lines)
